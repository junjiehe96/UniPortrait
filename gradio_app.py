import os
from io import BytesIO

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, AutoencoderKL, ControlNetModel, StableDiffusionControlNetPipeline
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from uniportrait import inversion
from uniportrait.uniportrait_attention_processor import attn_args
from uniportrait.uniportrait_pipeline import UniPortraitPipeline

port = 7860

device = "cuda"
torch_dtype = torch.float16

# base
base_model_path = "SG161222/Realistic_Vision_V5.1_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
controlnet_pose_ckpt = "lllyasviel/control_v11p_sd15_openpose"
# specific
image_encoder_path = "models/IP-Adapter/models/image_encoder"
ip_ckpt = "models/IP-Adapter/models/ip-adapter_sd15.bin"
face_backbone_ckpt = "models/glint360k_curricular_face_r101_backbone.bin"
uniportrait_faceid_ckpt = "models/uniportrait-faceid_sd15.bin"
uniportrait_router_ckpt = "models/uniportrait-router_sd15.bin"

# load controlnet
pose_controlnet = ControlNetModel.from_pretrained(controlnet_pose_ckpt, torch_dtype=torch_dtype)

# load SD pipeline
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch_dtype)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=[pose_controlnet],
    torch_dtype=torch_dtype,
    scheduler=noise_scheduler,
    vae=vae,
    # feature_extractor=None,
    # safety_checker=None,
)

# load uniportrait pipeline
uniportrait_pipeline = UniPortraitPipeline(pipe, image_encoder_path, ip_ckpt=ip_ckpt,
                                           face_backbone_ckpt=face_backbone_ckpt,
                                           uniportrait_faceid_ckpt=uniportrait_faceid_ckpt,
                                           uniportrait_router_ckpt=uniportrait_router_ckpt,
                                           device=device, torch_dtype=torch_dtype)

# load face detection assets
face_app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=["detection"])
face_app.prepare(ctx_id=0, det_size=(640, 640))


def pad_np_bgr_image(np_image, scale=1.25):
    assert scale >= 1.0, "scale should be >= 1.0"
    pad_scale = scale - 1.0
    h, w = np_image.shape[:2]
    top = bottom = int(h * pad_scale)
    left = right = int(w * pad_scale)
    ret = cv2.copyMakeBorder(np_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return ret, (left, top)


def process_faceid_image(pil_faceid_image):
    np_faceid_image = np.array(pil_faceid_image.convert("RGB"))
    img = cv2.cvtColor(np_faceid_image, cv2.COLOR_RGB2BGR)
    faces = face_app.get(img)  # bgr
    if len(faces) == 0:
        # padding, try again
        _h, _w = img.shape[:2]
        _img, left_top_coord = pad_np_bgr_image(img)
        faces = face_app.get(_img)
        if len(faces) == 0:
            gr.Info("Warning: No face detected in the image. Continue processing...")

        min_coord = np.array([0, 0])
        max_coord = np.array([_w, _h])
        sub_coord = np.array([left_top_coord[0], left_top_coord[1]])
        for face in faces:
            face.bbox = np.minimum(np.maximum(face.bbox.reshape(-1, 2) - sub_coord, min_coord), max_coord).reshape(4)
            face.kps = face.kps - sub_coord

    faces = sorted(faces, key=lambda x: abs((x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])), reverse=True)
    faceid_face = faces[0]
    norm_face = face_align.norm_crop(img, landmark=faceid_face.kps, image_size=224)
    pil_faceid_align_image = Image.fromarray(cv2.cvtColor(norm_face, cv2.COLOR_BGR2RGB))

    return pil_faceid_align_image


def prepare_single_faceid_cond_kwargs(pil_faceid_image=None, pil_faceid_supp_images=None,
                                      pil_faceid_mix_images=None, mix_scales=None):
    pil_faceid_align_images = []
    if pil_faceid_image:
        pil_faceid_align_images.append(process_faceid_image(pil_faceid_image))
    if pil_faceid_supp_images and len(pil_faceid_supp_images) > 0:
        for pil_faceid_supp_image in pil_faceid_supp_images:
            if isinstance(pil_faceid_supp_image, Image.Image):
                pil_faceid_align_images.append(process_faceid_image(pil_faceid_supp_image))
            else:
                pil_faceid_align_images.append(
                    process_faceid_image(Image.open(BytesIO(pil_faceid_supp_image)))
                )

    mix_refs = []
    mix_ref_scales = []
    if pil_faceid_mix_images:
        for pil_faceid_mix_image, mix_scale in zip(pil_faceid_mix_images, mix_scales):
            if pil_faceid_mix_image:
                mix_refs.append(process_faceid_image(pil_faceid_mix_image))
                mix_ref_scales.append(mix_scale)

    single_faceid_cond_kwargs = None
    if len(pil_faceid_align_images) > 0:
        single_faceid_cond_kwargs = {
            "refs": pil_faceid_align_images
        }
        if len(mix_refs) > 0:
            single_faceid_cond_kwargs["mix_refs"] = mix_refs
            single_faceid_cond_kwargs["mix_scales"] = mix_ref_scales

    return single_faceid_cond_kwargs


def text_to_single_id_generation_process(
        pil_faceid_image=None, pil_faceid_supp_images=None,
        pil_faceid_mix_image_1=None, mix_scale_1=0.0,
        pil_faceid_mix_image_2=None, mix_scale_2=0.0,
        faceid_scale=0.0, face_structure_scale=0.0,
        prompt="", negative_prompt="",
        num_samples=1, seed=-1,
        image_resolution="512x512",
        inference_steps=25,
):
    if seed == -1:
        seed = None

    single_faceid_cond_kwargs = prepare_single_faceid_cond_kwargs(pil_faceid_image,
                                                                  pil_faceid_supp_images,
                                                                  [pil_faceid_mix_image_1, pil_faceid_mix_image_2],
                                                                  [mix_scale_1, mix_scale_2])

    cond_faceids = [single_faceid_cond_kwargs] if single_faceid_cond_kwargs else []

    # reset attn args
    attn_args.reset()
    # set faceid condition
    attn_args.lora_scale = 1.0 if len(cond_faceids) == 1 else 0.0  # single-faceid lora
    attn_args.multi_id_lora_scale = 1.0 if len(cond_faceids) > 1 else 0.0  # multi-faceid lora
    attn_args.faceid_scale = faceid_scale if len(cond_faceids) > 0 else 0.0
    attn_args.num_faceids = len(cond_faceids)
    print(attn_args)

    h, w = int(image_resolution.split("x")[0]), int(image_resolution.split("x")[1])
    prompt = [prompt] * num_samples
    negative_prompt = [negative_prompt] * num_samples
    images = uniportrait_pipeline.generate(prompt=prompt, negative_prompt=negative_prompt,
                                           cond_faceids=cond_faceids, face_structure_scale=face_structure_scale,
                                           seed=seed, guidance_scale=7.5,
                                           num_inference_steps=inference_steps,
                                           image=[torch.zeros([1, 3, h, w])],
                                           controlnet_conditioning_scale=[0.0])
    final_out = []
    for pil_image in images:
        final_out.append(pil_image)

    for single_faceid_cond_kwargs in cond_faceids:
        final_out.extend(single_faceid_cond_kwargs["refs"])
        if "mix_refs" in single_faceid_cond_kwargs:
            final_out.extend(single_faceid_cond_kwargs["mix_refs"])

    return final_out


def text_to_multi_id_generation_process(
        pil_faceid_image_1=None, pil_faceid_supp_images_1=None,
        pil_faceid_mix_image_1_1=None, mix_scale_1_1=0.0,
        pil_faceid_mix_image_1_2=None, mix_scale_1_2=0.0,
        pil_faceid_image_2=None, pil_faceid_supp_images_2=None,
        pil_faceid_mix_image_2_1=None, mix_scale_2_1=0.0,
        pil_faceid_mix_image_2_2=None, mix_scale_2_2=0.0,
        faceid_scale=0.0, face_structure_scale=0.0,
        prompt="", negative_prompt="",
        num_samples=1, seed=-1,
        image_resolution="512x512",
        inference_steps=25,
):
    if seed == -1:
        seed = None

    faceid_cond_kwargs_1 = prepare_single_faceid_cond_kwargs(pil_faceid_image_1,
                                                             pil_faceid_supp_images_1,
                                                             [pil_faceid_mix_image_1_1,
                                                              pil_faceid_mix_image_1_2],
                                                             [mix_scale_1_1, mix_scale_1_2])
    faceid_cond_kwargs_2 = prepare_single_faceid_cond_kwargs(pil_faceid_image_2,
                                                             pil_faceid_supp_images_2,
                                                             [pil_faceid_mix_image_2_1,
                                                              pil_faceid_mix_image_2_2],
                                                             [mix_scale_2_1, mix_scale_2_2])
    cond_faceids = []
    if faceid_cond_kwargs_1 is not None:
        cond_faceids.append(faceid_cond_kwargs_1)
    if faceid_cond_kwargs_2 is not None:
        cond_faceids.append(faceid_cond_kwargs_2)

    # reset attn args
    attn_args.reset()
    # set faceid condition
    attn_args.lora_scale = 1.0 if len(cond_faceids) == 1 else 0.0  # single-faceid lora
    attn_args.multi_id_lora_scale = 1.0 if len(cond_faceids) > 1 else 0.0  # multi-faceid lora
    attn_args.faceid_scale = faceid_scale if len(cond_faceids) > 0 else 0.0
    attn_args.num_faceids = len(cond_faceids)
    print(attn_args)

    h, w = int(image_resolution.split("x")[0]), int(image_resolution.split("x")[1])
    prompt = [prompt] * num_samples
    negative_prompt = [negative_prompt] * num_samples
    images = uniportrait_pipeline.generate(prompt=prompt, negative_prompt=negative_prompt,
                                           cond_faceids=cond_faceids, face_structure_scale=face_structure_scale,
                                           seed=seed, guidance_scale=7.5,
                                           num_inference_steps=inference_steps,
                                           image=[torch.zeros([1, 3, h, w])],
                                           controlnet_conditioning_scale=[0.0])

    final_out = []
    for pil_image in images:
        final_out.append(pil_image)

    for single_faceid_cond_kwargs in cond_faceids:
        final_out.extend(single_faceid_cond_kwargs["refs"])
        if "mix_refs" in single_faceid_cond_kwargs:
            final_out.extend(single_faceid_cond_kwargs["mix_refs"])

    return final_out


def image_to_single_id_generation_process(
        pil_faceid_image=None, pil_faceid_supp_images=None,
        pil_faceid_mix_image_1=None, mix_scale_1=0.0,
        pil_faceid_mix_image_2=None, mix_scale_2=0.0,
        faceid_scale=0.0, face_structure_scale=0.0,
        pil_ip_image=None, ip_scale=1.0,
        num_samples=1, seed=-1, image_resolution="768x512",
        inference_steps=25,
):
    if seed == -1:
        seed = None

    single_faceid_cond_kwargs = prepare_single_faceid_cond_kwargs(pil_faceid_image,
                                                                  pil_faceid_supp_images,
                                                                  [pil_faceid_mix_image_1, pil_faceid_mix_image_2],
                                                                  [mix_scale_1, mix_scale_2])

    cond_faceids = [single_faceid_cond_kwargs] if single_faceid_cond_kwargs else []

    h, w = int(image_resolution.split("x")[0]), int(image_resolution.split("x")[1])

    # Image Prompt and Style Aligned
    if pil_ip_image is None:
        gr.Error("Please upload a reference image")
    attn_args.reset()
    pil_ip_image = pil_ip_image.convert("RGB").resize((w, h))
    zts = inversion.ddim_inversion(uniportrait_pipeline.pipe, np.array(pil_ip_image), "", inference_steps, 2)
    zT, inversion_callback = inversion.make_inversion_callback(zts, offset=0)

    # reset attn args
    attn_args.reset()
    # set ip condition
    attn_args.ip_scale = ip_scale if pil_ip_image else 0.0
    # set faceid condition
    attn_args.lora_scale = 1.0 if len(cond_faceids) == 1 else 0.0  # lora for single faceid
    attn_args.multi_id_lora_scale = 1.0 if len(cond_faceids) > 1 else 0.0  # lora for >1 faceids
    attn_args.faceid_scale = faceid_scale if len(cond_faceids) > 0 else 0.0
    attn_args.num_faceids = len(cond_faceids)
    # set shared self-attn
    attn_args.enable_share_attn = True
    attn_args.shared_score_shift = -0.5
    print(attn_args)

    prompt = [""] * (1 + num_samples)
    negative_prompt = [""] * (1 + num_samples)
    images = uniportrait_pipeline.generate(prompt=prompt, negative_prompt=negative_prompt,
                                           pil_ip_image=pil_ip_image,
                                           cond_faceids=cond_faceids, face_structure_scale=face_structure_scale,
                                           seed=seed, guidance_scale=7.5,
                                           num_inference_steps=inference_steps,
                                           image=[torch.zeros([1, 3, h, w])],
                                           controlnet_conditioning_scale=[0.0],
                                           zT=zT, callback_on_step_end=inversion_callback)
    images = images[1:]

    final_out = []
    for pil_image in images:
        final_out.append(pil_image)

    for single_faceid_cond_kwargs in cond_faceids:
        final_out.extend(single_faceid_cond_kwargs["refs"])
        if "mix_refs" in single_faceid_cond_kwargs:
            final_out.extend(single_faceid_cond_kwargs["mix_refs"])

    return final_out


def text_to_single_id_generation_block():
    gr.Markdown("## Text-to-Single-ID Generation")
    gr.HTML(text_to_single_id_description)
    gr.HTML(text_to_single_id_tips)
    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            prompt = gr.Textbox(value="", label='Prompt', lines=2)
            negative_prompt = gr.Textbox(value="nsfw", label='Negative Prompt')

            run_button = gr.Button(value="Run")
            with gr.Accordion("Options", open=True):
                image_resolution = gr.Dropdown(choices=["768x512", "512x512", "512x768"], value="512x512",
                                               label="Image Resolution (HxW)")
                seed = gr.Slider(label="Seed (-1 indicates random)", minimum=-1, maximum=2147483647, step=1,
                                 value=2147483647)
                num_samples = gr.Slider(label="Images", minimum=1, maximum=4, value=2, step=1)
                inference_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, visible=False)

                faceid_scale = gr.Slider(label="Face ID Scale", minimum=0.0, maximum=1.0, step=0.01, value=0.7)
                face_structure_scale = gr.Slider(label="Face Structure Scale", minimum=0.0, maximum=1.0,
                                                 step=0.01, value=0.1)

        with gr.Column(scale=2, min_width=100):
            with gr.Row(equal_height=False):
                pil_faceid_image = gr.Image(type="pil", label="ID Image")
                with gr.Accordion("ID Supplements", open=True):
                    with gr.Row():
                        pil_faceid_supp_images = gr.File(file_count="multiple", file_types=["image"],
                                                         type="binary", label="Additional ID Images")
                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            pil_faceid_mix_image_1 = gr.Image(type="pil", label="Mix ID 1")
                            mix_scale_1 = gr.Slider(label="Mix Scale 1", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                        with gr.Column(scale=1, min_width=100):
                            pil_faceid_mix_image_2 = gr.Image(type="pil", label="Mix ID 2")
                            mix_scale_2 = gr.Slider(label="Mix Scale 2", minimum=0.0, maximum=1.0, step=0.01, value=0.0)

            with gr.Row():
                example_output = gr.Image(type="pil", label="(Example Output)", visible=False)
                result_gallery = gr.Gallery(label='Output', show_label=True, elem_id="gallery", columns=4, preview=True,
                                            format="png")
    with gr.Row():
        examples = [
            [
                "A young man with short black hair, wearing a black hoodie with a hood, was paired with a blue denim jacket with yellow details.",
                "assets/examples/1-newton.jpg",
                "assets/examples/1-output-1.png",
            ],
        ]
        gr.Examples(
            label="Examples",
            examples=examples,
            fn=lambda x, y, z: (x, y),
            inputs=[prompt, pil_faceid_image, example_output],
            outputs=[prompt, pil_faceid_image]
        )
    ips = [
        pil_faceid_image, pil_faceid_supp_images,
        pil_faceid_mix_image_1, mix_scale_1,
        pil_faceid_mix_image_2, mix_scale_2,
        faceid_scale, face_structure_scale,
        prompt, negative_prompt,
        num_samples, seed,
        image_resolution,
        inference_steps,
    ]
    run_button.click(fn=text_to_single_id_generation_process, inputs=ips, outputs=[result_gallery])


def text_to_multi_id_generation_block():
    gr.Markdown("## Text-to-Multi-ID Generation")
    gr.HTML(text_to_multi_id_description)
    gr.HTML(text_to_multi_id_tips)
    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            prompt = gr.Textbox(value="", label='Prompt', lines=2)
            negative_prompt = gr.Textbox(value="nsfw", label='Negative Prompt')
            run_button = gr.Button(value="Run")
            with gr.Accordion("Options", open=True):
                image_resolution = gr.Dropdown(choices=["768x512", "512x512", "512x768"], value="512x512",
                                               label="Image Resolution (HxW)")
                seed = gr.Slider(label="Seed (-1 indicates random)", minimum=-1, maximum=2147483647, step=1,
                                 value=2147483647)
                num_samples = gr.Slider(label="Images", minimum=1, maximum=4, value=2, step=1)
                inference_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, visible=False)

                faceid_scale = gr.Slider(label="Face ID Scale", minimum=0.0, maximum=1.0, step=0.01, value=0.7)
                face_structure_scale = gr.Slider(label="Face Structure Scale", minimum=0.0, maximum=1.0,
                                                 step=0.01, value=0.3)

        with gr.Column(scale=2, min_width=100):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=100):
                    pil_faceid_image_1 = gr.Image(type="pil", label="First ID")
                    with gr.Accordion("First ID Supplements", open=False):
                        with gr.Row():
                            pil_faceid_supp_images_1 = gr.File(file_count="multiple", file_types=["image"],
                                                               type="binary", label="Additional ID Images")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                pil_faceid_mix_image_1_1 = gr.Image(type="pil", label="Mix ID 1")
                                mix_scale_1_1 = gr.Slider(label="Mix Scale 1", minimum=0.0, maximum=1.0, step=0.01,
                                                          value=0.0)
                            with gr.Column(scale=1, min_width=100):
                                pil_faceid_mix_image_1_2 = gr.Image(type="pil", label="Mix ID 2")
                                mix_scale_1_2 = gr.Slider(label="Mix Scale 2", minimum=0.0, maximum=1.0, step=0.01,
                                                          value=0.0)
                with gr.Column(scale=1, min_width=100):
                    pil_faceid_image_2 = gr.Image(type="pil", label="Second ID")
                    with gr.Accordion("Second ID Supplements", open=False):
                        with gr.Row():
                            pil_faceid_supp_images_2 = gr.File(file_count="multiple", file_types=["image"],
                                                               type="binary", label="Additional ID Images")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                pil_faceid_mix_image_2_1 = gr.Image(type="pil", label="Mix ID 1")
                                mix_scale_2_1 = gr.Slider(label="Mix Scale 1", minimum=0.0, maximum=1.0, step=0.01,
                                                          value=0.0)
                            with gr.Column(scale=1, min_width=100):
                                pil_faceid_mix_image_2_2 = gr.Image(type="pil", label="Mix ID 2")
                                mix_scale_2_2 = gr.Slider(label="Mix Scale 2", minimum=0.0, maximum=1.0, step=0.01,
                                                          value=0.0)

            with gr.Row():
                example_output = gr.Image(type="pil", label="(Example Output)", visible=False)
                result_gallery = gr.Gallery(label='Output', show_label=True, elem_id="gallery", columns=4, preview=True,
                                            format="png")
    with gr.Row():
        examples = [
            [
                "The two female models, fair-skinned, wore a white V-neck short-sleeved top with a light smile on the corners of their mouths. The background was off-white.",
                "assets/examples/2-stylegan2-ffhq-0100.png",
                "assets/examples/2-stylegan2-ffhq-0293.png",
                "assets/examples/2-output-1.png",
            ],
        ]
        gr.Examples(
            label="Examples",
            examples=examples,
            inputs=[prompt, pil_faceid_image_1, pil_faceid_image_2, example_output],
        )
    ips = [
        pil_faceid_image_1, pil_faceid_supp_images_1,
        pil_faceid_mix_image_1_1, mix_scale_1_1,
        pil_faceid_mix_image_1_2, mix_scale_1_2,
        pil_faceid_image_2, pil_faceid_supp_images_2,
        pil_faceid_mix_image_2_1, mix_scale_2_1,
        pil_faceid_mix_image_2_2, mix_scale_2_2,
        faceid_scale, face_structure_scale,
        prompt, negative_prompt,
        num_samples, seed,
        image_resolution,
        inference_steps,
    ]
    run_button.click(fn=text_to_multi_id_generation_process, inputs=ips, outputs=[result_gallery])


def image_to_single_id_generation_block():
    gr.Markdown("## Image-to-Single-ID Generation")
    gr.HTML(image_to_single_id_description)
    gr.HTML(image_to_single_id_tips)
    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            run_button = gr.Button(value="Run")
            seed = gr.Slider(label="Seed (-1 indicates random)", minimum=-1, maximum=2147483647, step=1,
                             value=2147483647)
            num_samples = gr.Slider(label="Images", minimum=1, maximum=4, value=2, step=1)
            image_resolution = gr.Dropdown(choices=["768x512", "512x512", "512x768"], value="512x512",
                                           label="Image Resolution (HxW)")
            inference_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, visible=False)

            ip_scale = gr.Slider(label="Reference Scale", minimum=0.0, maximum=1.0, step=0.01, value=0.7)
            faceid_scale = gr.Slider(label="Face ID Scale", minimum=0.0, maximum=1.0, step=0.01, value=0.7)
            face_structure_scale = gr.Slider(label="Face Structure Scale", minimum=0.0, maximum=1.0, step=0.01,
                                             value=0.3)

        with gr.Column(scale=3, min_width=100):
            with gr.Row(equal_height=False):
                pil_ip_image = gr.Image(type="pil", label="Portrait Reference")
                pil_faceid_image = gr.Image(type="pil", label="ID Image")
                with gr.Accordion("ID Supplements", open=True):
                    with gr.Row():
                        pil_faceid_supp_images = gr.File(file_count="multiple", file_types=["image"],
                                                         type="binary", label="Additional ID Images")
                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            pil_faceid_mix_image_1 = gr.Image(type="pil", label="Mix ID 1")
                            mix_scale_1 = gr.Slider(label="Mix Scale 1", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                        with gr.Column(scale=1, min_width=100):
                            pil_faceid_mix_image_2 = gr.Image(type="pil", label="Mix ID 2")
                            mix_scale_2 = gr.Slider(label="Mix Scale 2", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
            with gr.Row():
                with gr.Column(scale=3, min_width=100):
                    example_output = gr.Image(type="pil", label="(Example Output)", visible=False)
                    result_gallery = gr.Gallery(label='Output', show_label=True, elem_id="gallery", columns=4,
                                                preview=True, format="png")
    with gr.Row():
        examples = [
            [
                "assets/examples/3-style-1.png",
                "assets/examples/3-stylegan2-ffhq-0293.png",
                0.7,
                0.3,
                "assets/examples/3-output-1.png",
            ],
            [
                "assets/examples/3-style-1.png",
                "assets/examples/3-stylegan2-ffhq-0293.png",
                0.6,
                0.0,
                "assets/examples/3-output-2.png",
            ],
            [
                "assets/examples/3-style-2.jpg",
                "assets/examples/3-stylegan2-ffhq-0381.png",
                0.7,
                0.3,
                "assets/examples/3-output-3.png",
            ],
            [
                "assets/examples/3-style-3.jpg",
                "assets/examples/3-stylegan2-ffhq-0381.png",
                0.6,
                0.0,
                "assets/examples/3-output-4.png",
            ],
        ]
        gr.Examples(
            label="Examples",
            examples=examples,
            fn=lambda x, y, z, w, v: (x, y, z, w),
            inputs=[pil_ip_image, pil_faceid_image, faceid_scale, face_structure_scale, example_output],
            outputs=[pil_ip_image, pil_faceid_image, faceid_scale, face_structure_scale]
        )
    ips = [
        pil_faceid_image, pil_faceid_supp_images,
        pil_faceid_mix_image_1, mix_scale_1,
        pil_faceid_mix_image_2, mix_scale_2,
        faceid_scale, face_structure_scale,
        pil_ip_image, ip_scale,
        num_samples, seed, image_resolution,
        inference_steps,
    ]
    run_button.click(fn=image_to_single_id_generation_process, inputs=ips, outputs=[result_gallery])


if __name__ == "__main__":
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

    title = r"""
            <div style="text-align: center;">
                <h1> UniPortrait: A Unified Framework for Identity-Preserving Single- and Multi-Human Image Personalization </h1>
                <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                    <a href="https://arxiv.org/pdf/2408.05939"><img src="https://img.shields.io/badge/arXiv-2408.05939-red"></a>
                    &nbsp;
                    <a href='https://aigcdesigngroup.github.io/UniPortrait-Page/'><img src='https://img.shields.io/badge/Project_Page-UniPortrait-green' alt='Project Page'></a>
                    &nbsp;
                    <a href="https://github.com/junjiehe96/UniPortrait"><img src="https://img.shields.io/badge/Github-Code-blue"></a>
                </div>
                </br>
            </div>
        """

    title_description = r"""
        This is the <b>official 🤗 Gradio demo</b> for <a href='https://arxiv.org/pdf/2408.05939' target='_blank'><b>UniPortrait: A Unified Framework for Identity-Preserving Single- and Multi-Human Image Personalization</b></a>.<br>
        The demo provides three capabilities: text-to-single-ID personalization, text-to-multi-ID personalization, and image-to-single-ID personalization. All of these are based on the <b>Stable Diffusion v1-5</b> model. Feel free to give them a try! 😊
        """

    text_to_single_id_description = r"""🚀🚀🚀Quick start:<br>
        1. Enter a text prompt (Chinese or English), Upload an image with a face, and Click the <b>Run</b> button. 🤗<br>
        """

    text_to_single_id_tips = r"""💡💡💡Tips:<br>
        1. Try to avoid creating too small faces, as this may lead to some artifacts. (Currently, the short side length of the generated image is limited to 512)<br>
        2. It's a good idea to upload multiple reference photos of your face to improve the prompt and ID consistency. Additional references can be uploaded in the "ID supplements".<br>
        3. The appropriate values of "Face ID Scale" and "Face Structure Scale" are important for balancing the ID and text alignment. We recommend using "Face ID Scale" (0.5~0.7) and "Face Structure Scale" (0.0~0.4).<br>
        """

    text_to_multi_id_description = r"""🚀🚀🚀Quick start:<br>
        1. Enter a text prompt (Chinese or English), Upload an image with a face in "First ID" and "Second ID" blocks respectively, and Click the <b>Run</b> button. 🤗<br>
        """

    text_to_multi_id_tips = r"""💡💡💡Tips:<br>
        1. Try to avoid creating too small faces, as this may lead to some artifacts. (Currently, the short side length of the generated image is limited to 512)<br>
        2. It's a good idea to upload multiple reference photos of your face to improve the prompt and ID consistency. Additional references can be uploaded in the "ID supplements".<br>
        3. The appropriate values of "Face ID Scale" and "Face Structure Scale" are important for balancing the ID and text alignment. We recommend using "Face ID Scale" (0.3~0.7) and "Face Structure Scale" (0.0~0.4).<br>
        """

    image_to_single_id_description = r"""🚀🚀🚀Quick start: Upload an image as the portrait reference (can be any style), Upload a face image, and Click the <b>Run</b> button. 🤗<br>"""

    image_to_single_id_tips = r"""💡💡💡Tips:<br>
        1. Try to avoid creating too small faces, as this may lead to some artifacts. (Currently, the short side length of the generated image is limited to 512)<br>
        2. It's a good idea to upload multiple reference photos of your face to improve ID consistency. Additional references can be uploaded in the "ID supplements".<br>
        3. The appropriate values of "Face ID Scale" and "Face Structure Scale" are important for balancing the portrait reference and ID alignment. We recommend using "Face ID Scale" (0.5~0.7) and "Face Structure Scale" (0.0~0.4).<br>
        """

    citation = r"""
        ---
        📝 **Citation**
        <br>
        If our work is helpful for your research or applications, please cite us via:
        ```bibtex
        @article{he2024uniportrait,
          title={UniPortrait: A Unified Framework for Identity-Preserving Single-and Multi-Human Image Personalization},
          author={He, Junjie and Geng, Yifeng and Bo, Liefeng},
          journal={arXiv preprint arXiv:2408.05939},
          year={2024}
        }
        ```
        📧 **Contact**
        <br>
        If you have any questions, please feel free to open an issue or directly reach us out at <b>hejunjie1103@gmail.com</b>.
        """

    block = gr.Blocks(title="UniPortrait").queue()
    with block:
        gr.HTML(title)
        gr.HTML(title_description)

        with gr.TabItem("Text-to-Single-ID"):
            text_to_single_id_generation_block()

        with gr.TabItem("Text-to-Multi-ID"):
            text_to_multi_id_generation_block()

        with gr.TabItem("Image-to-Single-ID (Stylization)"):
            image_to_single_id_generation_block()

        gr.Markdown(citation)

    block.launch(server_name='0.0.0.0', share=False, server_port=port, allowed_paths=["/"])
