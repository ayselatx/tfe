import os
import glob
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_DIR = "/home/aysel/tfe/TFE_Data"
TFE_DATA = "/home/aysel/tfe/TFE_Data/TFE_Data"

DATASETS_DIR    = os.path.join(BASE_DIR, "Datasets")
UNIMODAL_DIR    = os.path.join(BASE_DIR, "Unimodal_Results")
PROJECTION_DIR  = os.path.join(BASE_DIR, "Projections")
ATTRIBUTION_DIR = os.path.join(BASE_DIR, "Attributions")
RETRIEVAL_DIR   = os.path.join(BASE_DIR, "Retrieval")
CLUSTER_DIR     = os.path.join(BASE_DIR, "Clusters")

# Flickr8k image directory
FLICKR8K_IMG_DIR = os.path.join(BASE_DIR, "Flickr8k", "Images", "Flicker8k_Dataset")
FLICKR8K_CAPTIONS = os.path.join(BASE_DIR, "Flickr8k", "Text", "Flickr8k.token.txt")

# All image paths (sorted by filename)
IMAGE_PATHS = sorted(
    glob.glob(os.path.join(FLICKR8K_IMG_DIR, "*.jpg")),
    key=lambda p: os.path.basename(p)
)

EMBED_DIR       = os.path.join(BASE_DIR, "Embeddings")
VISION_DIR      = os.path.join(EMBED_DIR, "vision")
TEXT_DIR        = os.path.join(EMBED_DIR, "text")

MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, "Model_Artifacts")
VISION_ARTIFACTS_DIR = os.path.join(MODEL_ARTIFACTS_DIR, "vision")
TEXT_ARTIFACTS_DIR = os.path.join(MODEL_ARTIFACTS_DIR, "text")

UNIMODAL_EVAL_DIR = os.path.join(BASE_DIR, "Evaluations", "Unimodal")
VISION_EVAL_DIR = os.path.join(UNIMODAL_EVAL_DIR, "vision")
TEXT_EVAL_DIR = os.path.join(UNIMODAL_EVAL_DIR, "text")

ATTRIB_DIR = os.path.join(MODEL_ARTIFACTS_DIR, "attributions")

VISION_XAI_DIR = os.path.join(BASE_DIR, "Attributions", "Vision")
TEXT_XAI_DIR = os.path.join(BASE_DIR, "Attributions", "Text")

PROJECTIONS_DIR = os.path.join("/home/aysel/tfe/TFE_Data/TFE_Data/Model_Artifacts/projections")
PROJECTIONS_DATA = "/home/aysel/tfe/TFE_Data/TFE_Data/Model_Artifacts/projections"

STREAMLIT_DATA = "/home/aysel/tfe/streamlit_app/data"
STREAMLIT_DATA_UNIMODAL = os.path.join(STREAMLIT_DATA, "unimodal")
STREAMLIT_DATA_VISION_DIR = os.path.join(STREAMLIT_DATA_UNIMODAL, "vision")
STREAMLIT_DATA_TEXT_DIR = os.path.join(STREAMLIT_DATA_UNIMODAL, "text")
STREAMLIT_DATASETS_DIR = os.path.join(STREAMLIT_DATA, "dataset")

def ensure_dirs():
    for d in [EMBED_DIR, VISION_DIR, TEXT_DIR,
              MODEL_ARTIFACTS_DIR, VISION_ARTIFACTS_DIR, TEXT_ARTIFACTS_DIR,
              UNIMODAL_EVAL_DIR, VISION_EVAL_DIR, TEXT_EVAL_DIR,
              VISION_XAI_DIR, TEXT_XAI_DIR,
              PROJECTIONS_DIR]:
        os.makedirs(d, exist_ok=True)       

def vision_emb_path(dataset, model):
    return os.path.join(VISION_DIR, dataset, f"{model}.npy")

def text_emb_path(dataset, model):
    return os.path.join(TEXT_DIR, dataset, f"{model}.npy")

def vision_artifact_dir(dataset, model):
    return os.path.join(VISION_ARTIFACTS_DIR, dataset, model)

def text_artifact_dir(dataset, model):
    return os.path.join(TEXT_ARTIFACTS_DIR, dataset, model)

def vision_eval_dir(dataset, model):
    return os.path.join(VISION_EVAL_DIR, dataset, model)

def text_eval_dir(dataset, model):
    return os.path.join(TEXT_EVAL_DIR, dataset, model)

def vision_xai_dir(dataset, model):
    return os.path.join(VISION_XAI_DIR, dataset, model)

def text_xai_dir(dataset, model):
    return os.path.join(TEXT_XAI_DIR, dataset, model)

def projection_dir(dataset, vision_model, text_model, method):
    return os.path.join(PROJECTIONS_DIR, dataset, f"{vision_model}_{text_model}", method)

def projection_matrix_path(dataset, vision_model, text_model, method, which):
    # which ∈ {"Wv", "Wt"}
    return os.path.join(projection_dir(dataset, vision_model, text_model, method), f"{which}.npy")

def projected_emb_path(dataset, vision_model, text_model, method, modality):
    # modality ∈ {"Xv", "Xt"}
    return os.path.join(projection_dir(dataset, vision_model, text_model, method), f"{modality}.npy")

def fusion_query_path(dataset, vision_model, text_model, method, fusion_op):
    return os.path.join(projection_dir(dataset, vision_model, text_model, method), f"fusion_{fusion_op}.npy")

def projection_metrics_path(dataset):
    return os.path.join(PROJECTIONS_DIR, dataset, "projection_metrics.csv")

def unimodal_attributions_path(dataset):
    return f"{ATTRIB_DIR}/{dataset}/captum_attributions.pkl"

def retrieval_attributions_path(dataset):
    return f"{ATTRIB_DIR}/{dataset}/captum_retrieval_attributions.pkl"



# from project_paths import vision_emb_path, text_emb_path, projection_path


"""
def vision_emb_path(dataset, model):
    return os.path.join(UNIMODAL_DIR, dataset, "vision", model, "embeddings.npy")

def text_emb_path(dataset, model):
    return os.path.join(UNIMODAL_DIR, dataset, "text", model, "embeddings.npy")

def projection_path(dataset, vmodel, tmodel):
    return os.path.join(PROJECTION_DIR, dataset, f"{vmodel}_{tmodel}_proj.npy")    
"""

"""
from project_paths import vision_emb_path
Xv = np.load(vision_emb_path("Flickr8k", "resnet50"))
"""
