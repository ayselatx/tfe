# utils/models_embedding.py

import torch
import streamlit as st
from transformers import (
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel,
    GPT2Tokenizer, GPT2Model,
    ViTModel, ViTImageProcessor,
)
from torchvision import models


# ---------------------------
# TEXT ENCODERS (cached)
# ---------------------------

@st.cache_resource
def load_bert_encoder(device):
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    try:
        model = model.to(device).eval()
    except RuntimeError:
        model = model.cpu().eval()
    return model, tok


@st.cache_resource
def load_roberta_encoder(device):
    tok = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")
    try:
        model = model.to(device).eval()
    except RuntimeError:
        model = model.cpu().eval()
    return model, tok


@st.cache_resource
def load_gpt2_encoder(device):
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2Model.from_pretrained("gpt2")
    try:
        model = model.to(device).eval()
    except RuntimeError:
        model = model.cpu().eval()
    return model, tok


# ---------------------------
# VISION ENCODERS (cached)
# ---------------------------

@st.cache_resource
def load_resnet50(device):
    model = models.resnet50(weights="DEFAULT")
    model.fc = torch.nn.Identity()
    try:
        model = model.to(device).eval()
    except RuntimeError:
        model = model.cpu().eval()
    return model


@st.cache_resource
def load_mobilenet_v3(device):
    model = models.mobilenet_v3_large(weights="DEFAULT")
    model.classifier = torch.nn.Identity()
    try:
        model = model.to(device).eval()
    except RuntimeError:
        model = model.cpu().eval()
    return model


@st.cache_resource
def load_vit(device):
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    try:
        model = model.to(device).eval()
    except RuntimeError:
        model = model.cpu().eval()
    return model, processor
