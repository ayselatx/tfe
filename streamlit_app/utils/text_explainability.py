import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.paths import text_xai_dir
from captum.attr import IntegratedGradients, Saliency

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache

# ---------------------------------------------------------
# TOKEN VISUALIZATION
# ---------------------------------------------------------
def visualize_tokens(tokenizer, input_ids, attributions, title="Token Attribution"):
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

    # Reduce attributions to 1D
    scores = attributions.squeeze()
    if scores.ndim > 1:
        scores = scores.sum(axis=-1)
    scores = scores.detach().cpu().numpy()

    # Normalize
    scores = scores / (np.max(np.abs(scores)) + 1e-8)

    # ⭐ FIX: sanitize NaN / inf
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)

    special = set(tokenizer.all_special_tokens)
    filtered_tokens, filtered_scores = [], []

    for tok, score in zip(tokens, scores):
        if tok in special or tok.strip() == "":
            continue
        tok = tok.replace("Ġ", "").replace("Ċ", "")
        filtered_tokens.append(tok)
        filtered_scores.append(score)

    # Adaptive figure width
    fig_width = min(max(len(filtered_tokens) * 0.5, 6), 18)
    fig, ax = plt.subplots(figsize=(fig_width, 1.8))
    ax.axis("off")

    renderer = fig.canvas.get_renderer()
    token_widths = []
    for tok in filtered_tokens:
        t = ax.text(0, 0, tok, fontsize=11)
        bb = t.get_window_extent(renderer=renderer)
        token_widths.append(bb.width / fig.dpi / fig.get_size_inches()[0])
        t.remove()

    total_width = sum(token_widths) + 0.01 * (len(filtered_tokens) - 1)
    start_x = max((1 - total_width) / 2, 0.02)
    x, y = start_x, 0.5

    for tok, score, width in zip(filtered_tokens, filtered_scores, token_widths):
        color = plt.cm.Reds(abs(score))  # now always valid
        ax.text(
            x, y, tok,
            fontsize=11,
            transform=ax.transAxes,
            bbox=dict(facecolor=color, alpha=0.8, edgecolor="black", boxstyle="round,pad=0.25")
        )
        x += width + 0.01

    plt.title(title)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------
# LOAD SAVED ATTRIBUTION RECORD
# -------------------------------------------------------
TEXT_MODEL_DIRNAMES = {
    "bert": "BERT",
    "roberta": "RoBERTa",
    "gpt2": "GPT2",
}


def load_text_explanation_record(model_name, caption_idx):
    dirname = TEXT_MODEL_DIRNAMES.get(model_name.lower(), model_name)
    master_path = os.path.join(text_xai_dir("Flickr8k", dirname), "captum_records.pkl")

    with open(master_path, "rb") as f:
        all_records = pickle.load(f)

    return all_records[int(caption_idx)]


class EmbeddingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, embeddings, attention_mask=None):
        out = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )
        return out.logits

def compute_text_attributions_on_the_fly(model, tokenizer, text):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )

    batch = {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device)
    }

    # Forward to get predicted label
    with torch.no_grad():
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        label = logits.argmax(dim=1).item()

    # Compute IG
    emb_layer = model.get_input_embeddings()
    input_embeds = emb_layer(batch["input_ids"])

    wrapper = EmbeddingWrapper(model)
    ig = IntegratedGradients(wrapper)

    attr = ig.attribute(
        input_embeds,
        additional_forward_args=(batch["attention_mask"],),
        target=label
    )

    return batch["input_ids"].cpu(), attr.cpu()
