import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ControlNetModel
from diffusers.pipelines.controlnet import MultiControlNetModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .curricular_face.backbone import get_model
from .resampler import UniPortraitFaceIDResampler
from .uniportrait_attention_processor import UniPortraitCNAttnProcessor2_0 as UniPortraitCNAttnProcessor
from .uniportrait_attention_processor import UniPortraitLoRAAttnProcessor2_0 as UniPortraitLoRAAttnProcessor
from .uniportrait_attention_processor import UniPortraitLoRAIPAttnProcessor2_0 as UniPortraitLoRAIPAttnProcessor


class ImageProjModel(nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds  # b, c
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens,
                                                              self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class UniPortraitPipeline:

    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt=None, face_backbone_ckpt=None, uniportrait_faceid_ckpt=None,
                 uniportrait_router_ckpt=None, num_ip_tokens=4, num_faceid_tokens=16,
                 lora_rank=128, device=torch.device("cuda"), torch_dtype=torch.float16):

        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.uniportrait_faceid_ckpt = uniportrait_faceid_ckpt
        self.uniportrait_router_ckpt = uniportrait_router_ckpt

        self.num_ip_tokens = num_ip_tokens
        self.num_faceid_tokens = num_faceid_tokens
        self.lora_rank = lora_rank

        self.device = device
        self.torch_dtype = torch_dtype

        self.pipe = sd_pipe.to(self.device)

        # load clip image encoder
        self.clip_image_processor = CLIPImageProcessor(size={"shortest_edge": 224}, do_center_crop=False,
                                                       use_square_size=True)
        self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=self.torch_dtype)
        # load face backbone
        self.facerecog_model = get_model("IR_101")([112, 112])
        self.facerecog_model.load_state_dict(torch.load(face_backbone_ckpt, map_location="cpu"))
        self.facerecog_model = self.facerecog_model.to(self.device, dtype=torch_dtype)
        self.facerecog_model.eval()
        # image proj model
        self.image_proj_model = self.init_image_proj()
        # faceid proj model
        self.faceid_proj_model = self.init_faceid_proj()
        # set uniportrait and ip adapter
        self.set_uniportrait_and_ip_adapter()
        # load uniportrait and ip adapter
        self.load_uniportrait_and_ip_adapter()

    def init_image_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.clip_image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_ip_tokens,
        ).to(self.device, dtype=self.torch_dtype)
        return image_proj_model

    def init_faceid_proj(self):
        faceid_proj_model = UniPortraitFaceIDResampler(
            intrinsic_id_embedding_dim=512,
            structure_embedding_dim=64 + 128 + 256 + self.clip_image_encoder.config.hidden_size,
            num_tokens=16, depth=6,
            dim=self.pipe.unet.config.cross_attention_dim, dim_head=64,
            heads=12, ff_mult=4,
            output_dim=self.pipe.unet.config.cross_attention_dim
        ).to(self.device, dtype=self.torch_dtype)
        return faceid_proj_model

    def set_uniportrait_and_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = UniPortraitLoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=self.lora_rank,
                ).to(self.device, dtype=self.torch_dtype).eval()
            else:
                attn_procs[name] = UniPortraitLoRAIPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=self.lora_rank,
                    num_ip_tokens=self.num_ip_tokens,
                    num_faceid_tokens=self.num_faceid_tokens,
                ).to(self.device, dtype=self.torch_dtype).eval()
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, ControlNetModel):
                self.pipe.controlnet.set_attn_processor(
                    UniPortraitCNAttnProcessor(
                        num_ip_tokens=self.num_ip_tokens,
                        num_faceid_tokens=self.num_faceid_tokens,
                    )
                )
            elif isinstance(self.pipe.controlnet, MultiControlNetModel):
                for module in self.pipe.controlnet.nets:
                    module.set_attn_processor(
                        UniPortraitCNAttnProcessor(
                            num_ip_tokens=self.num_ip_tokens,
                            num_faceid_tokens=self.num_faceid_tokens,
                        )
                    )
            else:
                raise ValueError

    def load_uniportrait_and_ip_adapter(self):
        if self.ip_ckpt:
            print(f"loading from {self.ip_ckpt}...")
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
            self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
            ip_layers = nn.ModuleList(self.pipe.unet.attn_processors.values())
            ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

        if self.uniportrait_faceid_ckpt:
            print(f"loading from {self.uniportrait_faceid_ckpt}...")
            state_dict = torch.load(self.uniportrait_faceid_ckpt, map_location="cpu")
            self.faceid_proj_model.load_state_dict(state_dict["faceid_proj"], strict=True)
            ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
            ip_layers.load_state_dict(state_dict["faceid_adapter"], strict=False)

            if self.uniportrait_router_ckpt:
                print(f"loading from {self.uniportrait_router_ckpt}...")
                state_dict = torch.load(self.uniportrait_router_ckpt, map_location="cpu")
                router_state_dict = {}
                for k, v in state_dict["faceid_adapter"].items():
                    if "lora." in k:
                        router_state_dict[k.replace("lora.", "multi_id_lora.")] = v
                    elif "router." in k:
                        router_state_dict[k] = v
                ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
                ip_layers.load_state_dict(router_state_dict, strict=False)

    @torch.inference_mode()
    def get_ip_embeds(self, pil_ip_image):
        ip_image = self.clip_image_processor(images=pil_ip_image, return_tensors="pt").pixel_values
        ip_image = ip_image.to(self.device, dtype=self.torch_dtype)  # (b, 3, 224, 224), values being normalized
        ip_embeds = self.clip_image_encoder(ip_image).image_embeds
        ip_prompt_embeds = self.image_proj_model(ip_embeds)
        uncond_ip_prompt_embeds = self.image_proj_model(torch.zeros_like(ip_embeds))
        return ip_prompt_embeds, uncond_ip_prompt_embeds

    @torch.inference_mode()
    def get_single_faceid_embeds(self, pil_face_images, face_structure_scale):
        face_clip_image = self.clip_image_processor(images=pil_face_images, return_tensors="pt").pixel_values
        face_clip_image = face_clip_image.to(self.device, dtype=self.torch_dtype)  # (b, 3, 224, 224)
        face_clip_embeds = self.clip_image_encoder(
            face_clip_image, output_hidden_states=True).hidden_states[-2][:, 1:]  # b, 256, 1280

        OPENAI_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device,
                                        dtype=self.torch_dtype).reshape(-1, 1, 1)
        OPENAI_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device,
                                       dtype=self.torch_dtype).reshape(-1, 1, 1)
        facerecog_image = face_clip_image * OPENAI_CLIP_STD + OPENAI_CLIP_MEAN  # [0, 1]
        facerecog_image = torch.clamp((facerecog_image - 0.5) / 0.5, -1, 1)  # [-1, 1]
        facerecog_image = F.interpolate(facerecog_image, size=(112, 112), mode="bilinear", align_corners=False)
        facerecog_embeds = self.facerecog_model(facerecog_image, return_mid_feats=True)[1]

        face_intrinsic_id_embeds = facerecog_embeds[-1]  # (b, 512, 7, 7)
        face_intrinsic_id_embeds = face_intrinsic_id_embeds.flatten(2).permute(0, 2, 1)  # b, 49, 512

        facerecog_structure_embeds = facerecog_embeds[:-1]  # (b, 64, 56, 56), (b, 128, 28, 28), (b, 256, 14, 14)
        facerecog_structure_embeds = torch.cat([
            F.interpolate(feat, size=(16, 16), mode="bilinear", align_corners=False)
            for feat in facerecog_structure_embeds], dim=1)  # b, 448, 16, 16
        facerecog_structure_embeds = facerecog_structure_embeds.flatten(2).permute(0, 2, 1)  # b, 256, 448
        face_structure_embeds = torch.cat([facerecog_structure_embeds, face_clip_embeds], dim=-1)  # b, 256, 1728

        uncond_face_clip_embeds = self.clip_image_encoder(
            torch.zeros_like(face_clip_image[:1]), output_hidden_states=True).hidden_states[-2][:, 1:]  # 1, 256, 1280
        uncond_face_structure_embeds = torch.cat(
            [torch.zeros_like(facerecog_structure_embeds[:1]), uncond_face_clip_embeds], dim=-1)  # 1, 256, 1728

        faceid_prompt_embeds = self.faceid_proj_model(
            face_intrinsic_id_embeds.flatten(0, 1).unsqueeze(0),
            face_structure_embeds.flatten(0, 1).unsqueeze(0),
            structure_scale=face_structure_scale,
        )  # [b, 16, 768]

        uncond_faceid_prompt_embeds = self.faceid_proj_model(
            torch.zeros_like(face_intrinsic_id_embeds[:1]),
            uncond_face_structure_embeds,
            structure_scale=face_structure_scale,
        )  # [1, 16, 768]

        return faceid_prompt_embeds, uncond_faceid_prompt_embeds

    def generate(
            self,
            prompt=None,
            negative_prompt=None,
            pil_ip_image=None,
            cond_faceids=None,
            face_structure_scale=0.0,
            seed=-1,
            guidance_scale=7.5,
            num_inference_steps=30,
            zT=None,
            **kwargs,
    ):
        """
        Args:
            prompt:
            negative_prompt:
            pil_ip_image:
            cond_faceids: [
                {
                    "refs": [PIL.Image] or PIL.Image,
                    (Optional) "mix_refs": [PIL.Image],
                    (Optional) "mix_scales": [float],
                },
                ...
            ]
            face_structure_scale:
            seed:
            guidance_scale:
            num_inference_steps:
            zT:
            **kwargs:
        Returns:
        """

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=True,
                negative_prompt=negative_prompt)
            num_prompts = prompt_embeds.shape[0]

            if pil_ip_image is not None:
                ip_prompt_embeds, uncond_ip_prompt_embeds = self.get_ip_embeds(pil_ip_image)
                ip_prompt_embeds = ip_prompt_embeds.repeat(num_prompts, 1, 1)
                uncond_ip_prompt_embeds = uncond_ip_prompt_embeds.repeat(num_prompts, 1, 1)
            else:
                ip_prompt_embeds = uncond_ip_prompt_embeds = \
                    torch.zeros_like(prompt_embeds[:, :1]).repeat(1, self.num_ip_tokens, 1)

            prompt_embeds = torch.cat([prompt_embeds, ip_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_ip_prompt_embeds], dim=1)

            if cond_faceids and len(cond_faceids) > 0:
                all_faceid_prompt_embeds = []
                all_uncond_faceid_prompt_embeds = []
                for curr_faceid_info in cond_faceids:
                    refs = curr_faceid_info["refs"]
                    faceid_prompt_embeds, uncond_faceid_prompt_embeds = \
                        self.get_single_faceid_embeds(refs, face_structure_scale)
                    if "mix_refs" in curr_faceid_info:
                        mix_refs = curr_faceid_info["mix_refs"]
                        mix_scales = curr_faceid_info["mix_scales"]

                        master_face_mix_scale = 1.0 - sum(mix_scales)
                        faceid_prompt_embeds = faceid_prompt_embeds * master_face_mix_scale
                        for mix_ref, mix_scale in zip(mix_refs, mix_scales):
                            faceid_mix_prompt_embeds, _ = self.get_single_faceid_embeds(mix_ref, face_structure_scale)
                            faceid_prompt_embeds = faceid_prompt_embeds + faceid_mix_prompt_embeds * mix_scale

                    all_faceid_prompt_embeds.append(faceid_prompt_embeds)
                    all_uncond_faceid_prompt_embeds.append(uncond_faceid_prompt_embeds)

                faceid_prompt_embeds = torch.cat(all_faceid_prompt_embeds, dim=1)
                uncond_faceid_prompt_embeds = torch.cat(all_uncond_faceid_prompt_embeds, dim=1)
                faceid_prompt_embeds = faceid_prompt_embeds.repeat(num_prompts, 1, 1)
                uncond_faceid_prompt_embeds = uncond_faceid_prompt_embeds.repeat(num_prompts, 1, 1)

                prompt_embeds = torch.cat([prompt_embeds, faceid_prompt_embeds], dim=1)
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_faceid_prompt_embeds], dim=1)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        if zT is not None:
            h_, w_ = kwargs["image"][0].shape[-2:]
            latents = torch.randn(num_prompts, 4, h_ // 8, w_ // 8, device=self.device, generator=generator,
                                  dtype=self.pipe.unet.dtype)
            latents[0] = zT
        else:
            latents = None

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            latents=latents,
            **kwargs,
        ).images

        return images
