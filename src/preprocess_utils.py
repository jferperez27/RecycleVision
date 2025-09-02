from PIL import Image
import numpy as np
from tensorflow import keras
from config import *

def preprocess_image(img_path) -> np.ndarray:
    """
    Predicts the class of a single image using the provided model.
    """
    img = Image.open(img_path)
    image = img.convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]))
    image_array = keras.preprocessing.image.img_to_array(image)

    return np.expand_dims(image_array, axis=0)