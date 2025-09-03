from pathlib import Path
SEED = 42
BATCH_SIZE = 32
EPOCHS = 1
IMG_SIZE = (512, 384)
OUTPUT_DIR = "models"
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
REPO = "garythung/trashnet"
FILE = "dataset-resized.zip"
TRAIN_RATIO = 0.8
MOVE = False
DATA_DIR = Path("data/raw/dataset-resized")
TRAIN_DIR = Path("data/processed/train")
TEST_DIR = Path("data/processed/test")
MODEL_NAME = "trash_classifier_model_v1.keras"