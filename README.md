# VertexRegen: Mesh Generation with Continuous Level of Detail

This is my re-implementation of ICCV 2025 paper **VertexRegen**.

### 🌐 [Project Page](https://vertexregen.github.io/) | 📄 [Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_VertexRegen_Mesh_Generation_with_Continuous_Level_of_Detail_ICCV_2025_paper.pdf) | 📚 [arXiv](https://arxiv.org/abs/2508.09062)

![Teaser](https://vertexregen.github.io/static/images/teaser.webp)

---

## ✅ Project Status

- [x] Data generation and tokenization
- [ ] Training and inference *(in progress)*

---


## 🧩 Tokenization Library

A standalone tokenization library is available on [PyPI](https://pypi.org/project/vertexregen-tokenizer/).

### Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Or install the tokenizer from source (requires `libcgal-dev` and related system packages):

```bash
pip install -e tokenizer/
```

## 📦 Data Preparation
You can generate vertex-split data using the demo **ShapeNet** dataset (converted from [MeshGPT](https://github.com/audi/MeshGPT)).  
Other datasets with compatible fields (`uid`, `vertices`, `faces`) are supported.

### 1. Generate vertex-split data

```bash
python -m scripts.create_dataset -o dataset/collapsed_shapenet_q256 -q 256
```

### 2. Run demo tokenization
```bash
python -m scripts.demo_tokenize -i dataset/collapsed_shapenet_q256/ -o demo
```

## 🧠 Training (Coming Soon)
Training is planned after CVPR 2026. The recommended datasets include `Objaverse` and `Objaverse-XL` for large-scale pretraining.

With the tokenization library, training follows a standard next-token prediction setup using a Transformer decoder (e.g., [OPT-350M](https://huggingface.co/facebook/opt-350m)).

If you are interested in contributing or have access to GPU resources, feel free to reach out for collaboration.

## 🪶 Citation
```
@InProceedings{Zhang_2025_ICCV_VertexRegen,
    author    = {Zhang, Xiang and Siddiqui, Yawar and Avetisyan, Armen and Xie, Chris and Engel, Jakob and Howard-Jenkins, Henry},
    title     = {VertexRegen: Mesh Generation with Continuous Level of Detail},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {12570-12580}
}
```