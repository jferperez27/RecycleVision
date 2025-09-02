import tensorflow as tf
from tensorflow import keras
from config import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

all_test_images = list(TEST_DIR.rglob("*.jpg"))
all_test_names = [img.parent.name for img in all_test_images]
MODEL_PATH = "models/trash_classifier_model_v1.keras"

def load_model(model_path: str) -> keras.Model:
    model = keras.models.load_model(model_path)
    return model

def preprocess_image(img_path) -> np.ndarray:
    """
    Predicts the class of a single image using the provided model.
    """
    img = Image.open(img_path)
    image = img.convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]))
    image_array = keras.preprocessing.image.img_to_array(image)

    return np.expand_dims(image_array, axis=0)

def predict(model: keras.Model, img: np.ndarray) -> int:
    predictions = model.predict(img)
    predicted_class = tf.argmax(predictions[0]).numpy()
    return predicted_class

if __name__ == "__main__":
    image_index = 1
    model = load_model(MODEL_PATH)
    image = preprocess_image(all_test_images[image_index])
    predicted_class = predict(model, image)
    predicted_class = CLASS_NAMES[predicted_class]
    print(f"Predicted class: {predicted_class}")
    print(f"Actual class: {all_test_names[image_index]}")
    plt.imshow(plt.imread(all_test_images[image_index]))
    plt.title(f"Test image - Class: {all_test_names[image_index]}")
    plt.xlabel(f"Predicted class: {predicted_class} || Actual class: {all_test_names[image_index]}")
    plt.show()