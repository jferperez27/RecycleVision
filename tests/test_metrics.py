from tensorflow import keras
from src.config import CLASS_NAMES, TEST_DIR
from src.metrics import model_accuracy
from pathlib import Path

all_test_images = list(TEST_DIR.rglob("*.jpg"))
all_test_names = [img.parent.name for img in all_test_images]

def test_model_accuracy():
    model_path = Path("models/trash_classifier_model_v3.keras")

    all_test_images = list(TEST_DIR.rglob("*.jpg"))
    all_test_names = [img.parent.name for img in all_test_images]

    model = keras.models.load_model(model_path)
    accuracy = model_accuracy(model, all_test_images, all_test_names, CLASS_NAMES)
    assert 0.866 <= accuracy <= 1.0  # Accuracy should be between 86% - 100%

