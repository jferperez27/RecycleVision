import pytest
import tensorflow as tf
from tensorflow import keras
from src.config import *

all_test_images = list(TEST_DIR.rglob("*.jpg"))
all_test_names = [img.parent.name for img in all_test_images]

def test_model_predicts():
    model = keras.models.load_model("models/trash_classifier_model_v1.keras")
    dummy_input = tf.random.uniform((1, 512, 384, 3))
    predictions = model.predict(dummy_input)
    assert predictions.shape == (1, len(CLASS_NAMES))

if __name__ == "__main__":
    test_model_predicts()