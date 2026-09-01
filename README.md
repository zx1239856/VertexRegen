# VertexRegen: Mesh Generation with Continuous Level of Detail

[![Project Page](https://img.shields.io/badge/🌐-Project_Page-blue)](https://vertexregen.github.io/)
[![Paper](https://img.shields.io/badge/📄-Paper-green)](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_VertexRegen_Mesh_Generation_with_Continuous_Level_of_Detail_ICCV_2025_paper.pdf)
[![arXiv](https://img.shields.io/badge/📚-arXiv-b31b1b)](https://arxiv.org/abs/2508.09062)
[![PyPI](https://img.shields.io/badge/PyPI-vertexregen--tokenizer-blue)](https://pypi.org/project/vertexregen-tokenizer/)

**[Xiang Zhang](https://xzhang.dev)**, Yawar Siddiqui, Armen Avetisyan, Chris Xie, Jakob Engel, Henry Howard-Jenkins

*ICCV 2025*

> **Author's re-implementation.** This repository is written and maintained by the paper's first author. The original research implementation is not publicly available, so this independent re-implementation exists to keep the method usable by the community.

![Teaser](https://vertexregen.github.io/static/images/teaser.webp)

---

## 📖 Overview

**VertexRegen** is a mesh generation framework that supports **continuous levels of detail (LoD)**, enabling smooth transitions across mesh resolutions. This repository provides an end-to-end re-implementation of the method, including:

* Dataset preprocessing and vertex-split generation
* Mesh tokenization
* Model training
* Distributed inference

The goal of this project is to offer a clean, reproducible research codebase and practical tooling for experimenting with continuous-LoD mesh generation.

---

## 🚧 Project Status

* [x] Data generation & vertex-split preprocessing
* [x] Tokenization pipeline
* [x] Training & inference code
* [ ] Pre-trained weights on Objaverse / Objaverse-XL

> ⚠️ **Note**: Only overfitting configurations are currently released to validate the training and inference pipelines.

---

## 🧩 Tokenization Library

A standalone **VertexRegen tokenizer** is available on PyPI:

👉 [https://pypi.org/project/vertexregen-tokenizer/](https://pypi.org/project/vertexregen-tokenizer/)

You can either install it directly from PyPI or build it from source.

### Install from source

> Requires `libcgal-dev` and related CGAL system dependencies.

```bash
pip install -e tokenizer/
```

---

## 🛠️ Installation

### Python environment

Please ensure you have **PyTorch ≥ 2.9.1**, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

### Docker (recommended)

A pre-built Docker image is provided for convenience:

```bash
docker pull zx1239856/vertexregen:0.1.0
```

---

## 📦 Data Preparation

You can generate vertex-split datasets using the provided demo **ShapeNet** data (converted from MeshGPT format). Other datasets are also supported as long as they contain the following fields:

* `uid`
* `vertices`
* `faces`

### 1️⃣ Generate vertex-split data

```bash
python -m scripts.create_dataset -o dataset/collapsed_shapenet_q128 -q 128
```

### 2️⃣ Run demo tokenization

```bash
python -m scripts.demo_tokenize -i dataset/collapsed_shapenet_q128/ -o demo
```

---

## 🧠 Training & Inference

### Training

```bash
python launch.py --num-processes=4 train.py --config-name=shapenet_uncond_opt_350_overfit
```

### Inference (distributed)

Replace the model path with your actual checkpoint directory. Use `-h` for additional options.

```bash
python launch.py --num-processes=4 infer.py --model outputs/.../checkpoint-500/
```

> ℹ️ **Reproducibility note**: Fully reproducing the results from the paper requires training on large-scale datasets such as **Objaverse-XL**, which are not included in this release.

---
## 🏷️ License
This repository is released under the [MIT](LICENSE) license.


## 🪶 Citation

If you find this repository useful, please cite the paper:

```bibtex
@InProceedings{Zhang_2025_ICCV_VertexRegen,
    author    = {Zhang, Xiang and Siddiqui, Yawar and Avetisyan, Armen and Xie, Chris and Engel, Jakob and Howard-Jenkins, Henry},
    title     = {VertexRegen: Mesh Generation with Continuous Level of Detail},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {12570--12580}
}
```
