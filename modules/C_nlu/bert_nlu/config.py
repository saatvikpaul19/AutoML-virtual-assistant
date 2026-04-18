import os

# Base directory for bert_nlu
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "joint_distilbert_nlu.pt")
LABEL_MAPS_PATH = os.path.join(BASE_DIR, "models", "label_maps.pt")

CONFIDENCE_THRESHOLD = 0.7
USE_FALLBACK = True
