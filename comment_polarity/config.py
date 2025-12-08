from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "comments.csv"

HF_MODEL_NAME = "distilbert-base-uncased"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RANDOM_STATE = 42

# NEW: paths for saved models
MODELS_DIR = PROJECT_ROOT / "saved_models"
CLASSIC_MODEL_PATH = MODELS_DIR / "classic_tfidf_logreg.joblib"
NEURAL_MODEL_DIR = MODELS_DIR / "neural_distilbert"
