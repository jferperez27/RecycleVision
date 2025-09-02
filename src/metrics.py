import tensorflow as tf
from tensorflow import keras
import numpy as np
from src.preprocess_utils import preprocess_image

def model_accuracy(model: keras.Model, test_images: list, test_names: list, class_names: list, batch_size=32) -> float:
    X = np.concatenate([preprocess_image(p) for p in test_images], axis=0)
    predictions = model.predict(X, batch_size=batch_size, verbose=0)

    pred_idx = predictions.argmax(axis=1)

    idx_map = {name: i for i, name in enumerate(class_names)}
    actual_idx = np.array([idx_map[name] for name in test_names], dtype=np.int64)

    return float((pred_idx == actual_idx).mean())