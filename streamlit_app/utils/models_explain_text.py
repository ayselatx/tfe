# utils/models_explainability.py

import torch
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2Tokenizer,
    GPT2ForSequenceClassification
)
import torch.nn as nn


# ---------------------------------------------------------
# TEXT EXPLAINABILITY MODELS (backbone-based)
# ---------------------------------------------------------

@st.cache_resource
def load_bert(device):
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )
    try:
        model = model.to(device).eval()
    except RuntimeError:
        model = model.cpu().eval()
    return model, tok, model.bert


@st.cache_resource
def load_roberta(device):
    tok = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2
    )
    try:
        model = model.to(device).eval()
    except RuntimeError:
        model = model.cpu().eval()
    return model, tok, model.roberta


@st.cache_resource
def load_gpt2(device):
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2",
        num_labels=2,
        pad_token_id=tok.pad_token_id
    )
    try:
        model = model.to(device).eval()
    except RuntimeError:
        model = model.cpu().eval()

    return model, tok, model.transformer
