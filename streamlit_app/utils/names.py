def normalize_model_name(name: str) -> str:
    name = name.lower().strip()

    mapping = {
        "resnet50": "ResNet50",
        "resnet_50": "ResNet50",
        "mobilenetv3": "MobileNetV3",
        "mobilenet_v3": "MobileNetV3",
        "mobile_net_v3": "MobileNetV3",
        "pvt": "PVT",
        "pvtv2": "PVT",
        "vit": "ViT",
        "bert": "BERT",
        "roberta": "RoBERTa",
        "gpt2": "GPT2",
        "gpt-2": "GPT2",
    }

    return mapping.get(name, name)
