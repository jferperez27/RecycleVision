import tensorflow as tf
from tensorflow import keras

def test_model_loads():
    model = keras.models.load_model("models/trash_classifier_model_v3.keras")
    assert isinstance(model, tf.keras.Model)

def test_model_predicts():
    model = keras.models.load_model("models/trash_classifier_model_v3.keras")
    dummy_input = tf.random.uniform((1, 512, 384, 3))
    predictions = model.predict(dummy_input)
    assert predictions.shape == (1, 6)  # Assuming 6 classes as per CLASS_NAMES


if __name__ == "__main__":
    test_model_loads()
    test_model_predicts()