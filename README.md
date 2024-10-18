<div align="center">
<h1>UniPortrait: A Unified Framework for Identity-Preserving Single- and Multi-Human Image Personalization</h1>

<a href='https://aigcdesigngroup.github.io/UniPortrait-Page/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2408.05939'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
<a href='https://huggingface.co/spaces/Junjie96/UniPortrait'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>

</div>

<img src='assets/highlight.png'>



UniPortrait is an innovative human image personalization framework. It customizes single- and multi-ID images in a unified manner, providing high-fidelity identity preservation, extensive facial editability, free-form text description, and no requirement for a predetermined layout.

---

## Release

- [2024/10/18] 🔥 We release the inference code and demo, which has naively integrated [ControlNet](https://github.com/lllyasviel/ControlNet), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), and [StyleAligned](https://github.com/google/style-aligned). The weight for this version is consistent with the huggingface space and experiments in the paper. We are now working on generalizing our method to more advanced diffusion models and more general customized concepts. Please stay tuned!
- [2024/08/12] 🔥 We release the [technical report](https://arxiv.org/abs/2408.05939), [project page](https://aigcdesigngroup.github.io/UniPortrait-Page/), and [HuggingFace demo](https://huggingface.co/spaces/Junjie96/UniPortrait) 🤗!

## Quickstart
```shell
# Clone repository
git clone https://github.com/junjiehe96/UniPortrait.git

# install requirements
cd UniPortrait
pip install -r requirements.txt

# download the models
git lfs install
git clone https://huggingface.co/Junjie96/UniPortrait models
# download ip-adapter models 
# Note: recommend downloading manually. We do not require all IP adapter models.
git clone https://huggingface.co/h94/IP-Adapter models/IP-Adapter

# then you can use the gradio app
python gradio_app.py
```


## Applications
<img src='assets/application.png'>



## Cite
If you find UniPortrait useful for your research and applications, please cite us using this BibTeX:

```bibtex
@article{he2024uniportrait,
    title={UniPortrait: A Unified Framework for Identity-Preserving Single-and Multi-Human Image Personalization},
    author={He, Junjie and Geng, Yifeng and Bo, Liefeng},
    journal={arXiv preprint arXiv:2408.05939},
    year={2024}
}
```

For any question, please feel free to open an issue or contact us via hejunjie1103@gmail.com.
