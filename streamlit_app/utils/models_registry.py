# utils/models_registry.py

from utils.models_embedding import (
    load_bert_encoder, load_roberta_encoder, load_gpt2_encoder,
    load_resnet50, load_mobilenet_v3, load_vit
)

from utils.models_explain_text import (
    load_bert, load_roberta,
    load_gpt2
)

from utils.models_explain_vision import (
    load_resnet50_explain, load_mobilenet_v3_explain,
    load_pvt_explain, load_vit_explain
)

TEXT_EMBEDDERS = {
    "bert": load_bert_encoder,
    "roberta": load_roberta_encoder,
    "gpt2": load_gpt2_encoder,
}

TEXT_EXPLAIN_MODELS = {
    #"bert": load_bert,
    #"roberta": load_roberta,
    #"gpt2": load_gpt2,
    "BERT": load_bert,
    "RoBERTa": load_roberta,
    "GPT2": load_gpt2,
}

VISION_EMBEDDERS = {
    "resnet50": load_resnet50,
    "mobilenet_v3": load_mobilenet_v3,
    "vit": load_vit,
}

VISION_EXPLAIN_MODELS = {
    "resnet50": load_resnet50_explain,
    "mobilenet_v3": load_mobilenet_v3_explain,
    "vit": load_vit_explain,
    "pvt": load_pvt_explain,
    "ResNet50": load_resnet50_explain,
    "MobileNetV3": load_mobilenet_v3_explain,
    "ViT": load_vit_explain,
    "PVT": load_pvt_explain,
}
