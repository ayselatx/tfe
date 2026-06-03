import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import spacy

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------------------------------------------------------
# Load FLAN‑T5 RAG Model
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
rag_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device).eval()


# ---------------------------------------------------------
# RAG Refinement
# ---------------------------------------------------------
def build_rag_prompt(captions):
    return (
        "Rewrite the following captions into a single, coherent, human-like description.\n\n"
        + "\n".join([f"- {c}" for c in captions])
        + "\n\nRefined caption:"
    )


def rag_refine(captions, prompt):
    full_prompt = (
        prompt + "\n\n" +
        "\n".join([f"- {c}" for c in captions]) +
        "\n\nRefined caption:"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    output = rag_model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        temperature=0.7
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ---------------------------------------------------------
# TOP‑50 RETRIEVAL (Xv, Xt already provided)
# ---------------------------------------------------------
def get_top50(Xv, Xt, image_index, captions):
    sims = cosine_similarity(Xv[image_index].reshape(1, -1), Xt)[0]
    idx = np.argsort(sims)[::-1][:50]
    return [captions[i] for i in idx], Xt[idx], Xv[image_index]


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
def cosine_vec(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def evaluate_cosine(image_emb, top50_embs, refined_emb):
    orig = np.mean([cosine_vec(image_emb, e) for e in top50_embs])
    ref = cosine_vec(image_emb, refined_emb)
    return orig, ref


def evaluate_centroid_similarity(top50_embs, refined_emb):
    centroid = np.mean(top50_embs, axis=0)
    dist_orig = np.mean([np.linalg.norm(e - centroid) for e in top50_embs])
    dist_ref = np.linalg.norm(refined_emb - centroid)
    return dist_orig, dist_ref


def caption_entropy(captions):
    words = " ".join(captions).split()
    _, counts = np.unique(words, return_counts=True)
    return entropy(counts)


def evaluate_entropy(top50, refined):
    return caption_entropy(top50), caption_entropy([refined])


def compute_bleu(gt_caps, refined_caption):
    smoothie = SmoothingFunction().method4
    refs = [nltk.word_tokenize(cap.lower()) for cap in gt_caps]
    hyp  = nltk.word_tokenize(refined_caption.lower())
    return sentence_bleu(refs, hyp, smoothing_function=smoothie)


# ---------------------------------------------------------
# SEMANTIC DIFFERENCES
# ---------------------------------------------------------
nlp = spacy.load("en_core_web_sm")


def extract_concepts(text):
    doc = nlp(text.lower())
    nouns = {t.lemma_ for t in doc if t.pos_ == "NOUN"}
    verbs = {t.lemma_ for t in doc if t.pos_ == "VERB"}
    adjs  = {t.lemma_ for t in doc if t.pos_ == "ADJ"}
    return nouns, verbs, adjs


def compare_captions(gt_caps, refined_caption):
    gt_text = " ".join(gt_caps)
    gt_n, gt_v, gt_a = extract_concepts(gt_text)
    ref_n, ref_v, ref_a = extract_concepts(refined_caption)

    return {
        "missing_nouns": list(gt_n - ref_n),
        "added_nouns":   list(ref_n - gt_n),
        "missing_verbs": list(gt_v - ref_v),
        "added_verbs":   list(ref_v - gt_v),
        "missing_adjs":  list(gt_a - ref_a),
        "added_adjs":    list(ref_a - gt_a),
    }


# ---------------------------------------------------------
# FULL RAG EVALUATION (Xv, Xt provided by Streamlit)
# ---------------------------------------------------------
def evaluate_rag(
    Xv,
    Xt,
    image_index,
    captions,
    gt_caps,
    prompt,
    img_emb=None,
    top50_caps=None,
    top50_embs=None
):
    """
    Unified RAG evaluation for dataset + uploaded images.

    Dataset image:
        - image_index is provided
        - img_emb, top50_caps, top50_embs are computed internally

    Uploaded image:
        - image_index is None
        - img_emb, top50_caps, top50_embs MUST be provided
    """

    # ---------------------------------------------------------
    # 1. RETRIEVAL
    # ---------------------------------------------------------
    if image_index is not None:
        # Dataset image → use precomputed embeddings
        top50_caps, top50_embs, img_emb = get_top50(Xv, Xt, image_index, captions)

    else:
        # Uploaded image → ensure required inputs exist
        if img_emb is None or top50_caps is None or top50_embs is None:
            raise ValueError("Uploaded image requires img_emb, top50_caps, top50_embs")

    # ---------------------------------------------------------
    # 2. LLM REFINEMENT
    # ---------------------------------------------------------
    refined_caption = rag_refine(top50_caps, prompt)

    # ---------------------------------------------------------
    # 3. EMBED REFINED CAPTION
    # ---------------------------------------------------------
    if refined_caption in captions:
        idx = captions.index(refined_caption)
        refined_emb = Xt[idx]
    else:
        refined_emb = np.mean(top50_embs, axis=0)

    # ---------------------------------------------------------
    # 4. METRICS
    # ---------------------------------------------------------
    cos_o, cos_r = evaluate_cosine(img_emb, top50_embs, refined_emb)
    cent_o, cent_r = evaluate_centroid_similarity(top50_embs, refined_emb)
    ent_o, ent_r = evaluate_entropy(top50_caps, refined_caption)

    bleu = compute_bleu(gt_caps, refined_caption) if gt_caps else 0.0

    diff = compare_captions(gt_caps, refined_caption) if gt_caps else {
        "missing_nouns": [],
        "added_nouns": [],
        "missing_verbs": [],
        "added_verbs": [],
        "missing_adjs": [],
        "added_adjs": []
    }

    # ---------------------------------------------------------
    # 5. RETURN RESULTS
    # ---------------------------------------------------------
    return {
        "top50": top50_caps,
        "refined": refined_caption,
        "semantic_diff": diff,
        "metrics": {
            "cosine_original": cos_o,
            "cosine_refined": cos_r,
            "centroid_original": cent_o,
            "centroid_refined": cent_r,
            "entropy_original": ent_o,
            "entropy_refined": ent_r,
            "bleu": bleu
        }
    }
