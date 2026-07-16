from pathlib import Path

# ==========================
# Project Paths
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# ==========================
# File Paths
# ==========================

DATA_PATH = DATA_DIR / "titanic.csv"

LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic.pkl"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest.pkl"
PIPELINE_MODEL_PATH = MODELS_DIR / "pipeline.pkl"
PRODUCTION_MODEL_PATH = MODELS_DIR / "production_model.pkl"

# ==========================
# Dataset
# ==========================

TARGET = "Survived"

FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone",
    "Title"
]

NUMERIC_FEATURES = [
        "Pclass",
        "Age",
        "Fare",
        "FamilySize",
        "IsAlone"
]

CATEGORICAL_FEATURES = [
        "Sex",
        "Title"
]

# ==========================
# Train/Test
# ==========================

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 1000