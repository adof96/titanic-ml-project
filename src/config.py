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


# ==========================
# Train/Test
# ==========================

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ==========================
# Model Paths
# ==========================

MODEL_DIR = "models"

LOGISTIC_MODEL_PATH = "models/logistic.pkl"

RANDOM_FOREST_MODEL_PATH = "models/random_forest.pkl"

PIPELINE_MODEL_PATH = "models/pipeline.pkl"