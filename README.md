# Vision Transformer Zoo

A clean, extensible, and reusable factory for creating HuggingFace-based Vision Transformer models - including **ViT**, **DeiT**, **DINO**, **DINOv2**, and **CLIP Vision** — with optional dropout, freezing, and head configuration.

---

## Features

- Easy model construction via `build_model(...)`
- Support for pretrained HuggingFace models
- Pluggable classification head
- Automatic dropout configuration
- Backbone freezing support
- Model registry for easy extensibility

---

## Installation

### Local development install

```bash
git clone git@github.com:jbindaAI/vit_zoo.git
cd vit_zoo
pip install -e .
