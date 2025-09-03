import tensorflow as tf
from tensorflow import keras
from config import *
import numpy as np
import matplotlib.pyplot as plt
from preprocess_utils import preprocess_image
import random

all_test_images = list(TEST_DIR.rglob("*.jpg"))
all_test_names = [img.parent.name for img in all_test_images]
MODEL_PATH = "models/trash_classifier_model_v3.keras"

def update():
    global all_test_images, all_test_names
    all_test_images = list(TEST_DIR.rglob("*.jpg"))
    all_test_names = [img.parent.name for img in all_test_images]

def load_model(model_path: str) -> keras.Model:
    update()
    model = keras.models.load_model(model_path)
    return model

def predict(model: keras.Model, img: np.ndarray) -> int:
    predictions = model.predict(img)
    predicted_class = tf.argmax(predictions[0]).numpy()
    return predicted_class

def simple_model_predict():
    image_index = random.randrange(len(all_test_images))
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

def model_inference(image_path: str):
    model = load_model(MODEL_PATH)
    image = preprocess_image(image_path)
    predicted_class = predict(model, image)
    predicted_class = CLASS_NAMES[predicted_class]
    print(f"Predicted class: {predicted_class}")
    plt.imshow(plt.imread(image_path))
    plt.title(f"Inference image - Predicted class: {predicted_class}")
    plt.xlabel(f"Predicted class: {predicted_class}")
    plt.show()

def custom_model_test_inference(model : keras.Model):
    image_index = random.randrange(len(all_test_images))
    image = preprocess_image(all_test_images[image_index])
    predicted_class = predict(model, image)
    predicted_class = CLASS_NAMES[predicted_class]
    print(f"Predicted class: {predicted_class}")
    print(f"Actual class: {all_test_names[image_index]}")
    plt.imshow(plt.imread(all_test_images[image_index]))
    plt.title(f"Test image - Class: {all_test_names[image_index]}")
    plt.xlabel(f"Predicted class: {predicted_class} || Actual class: {all_test_names[image_index]}")
    plt.show()

if __name__ == "__main__":
    simple_model_predict()