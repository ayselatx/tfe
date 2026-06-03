# utils/models_explain_vision.py

import torch
import torch.nn as nn
import streamlit as st
from torchvision import models, transforms as T
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    ViTModel
)
from PIL import Image


# ---------------------------------------------------------
# RESNET50 (Explainability)
# ---------------------------------------------------------

@st.cache_resource
def load_resnet50_explain(device):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    try:
        model = model.to(device).eval()
    except RuntimeError:
        model = model.cpu().eval()
    transform = weights.transforms()
    return model, transform


# ---------------------------------------------------------
# MOBILENET V3 (Explainability)
# ---------------------------------------------------------

@st.cache_resource
def load_mobilenet_v3_explain(device):
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)
    try:
        model = model.to(device).eval()
    except RuntimeError:
        model = model.cpu().eval()
    transform = weights.transforms()
    return model, transform


# ---------------------------------------------------------
# VIT (Explainability)
# ---------------------------------------------------------

class ViTWithHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.head = nn.Linear(self.backbone.config.hidden_size, 1000)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)


@st.cache_resource
def load_vit_explain(device):
    model = ViTWithHead()
    try:
        model = model.to(device).eval()
    except RuntimeError:
        model = model.cpu().eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

    return model, transform


# ---------------------------------------------------------
# PVT (Explainability)
# ---------------------------------------------------------

class CaptumWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(pixel_values=x)
        return out.logits


@st.cache_resource
def load_pvt_explain(device):
    processor = AutoImageProcessor.from_pretrained("Zetatech/pvt-tiny-224")
    base_model = AutoModelForImageClassification.from_pretrained(
        "Zetatech/pvt-tiny-224"
    )

    model = CaptumWrapper(base_model)
    try:
        model = model.to(device).eval()
    except RuntimeError:
        model = model.cpu().eval()

    def transform(img):
        return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

    return model, transform
