import os, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dataloader import *
import matplotlib.pyplot as plt
from pathlib import Path
from config import *


all_test_images = list(TEST_DIR.rglob("*.jpg"))
all_test_names = [img.parent.name for img in all_test_images]
all_train_images = list(TRAIN_DIR.rglob("*.jpg"))
all_train_names = [img.parent.name for img in all_train_images]

test_img = random.choice(all_test_images)
train_img = random.choice(all_train_images)
plt.imshow(plt.imread(all_test_images[0]))
plt.axis("off")
plt.title(f"Test image - Class: {all_test_names[0]}")
plt.show()

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_aug")

base = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=EPOCHS, 
                    validation_data=val_ds)

test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print(test_acc)

model.save(os.path.join(OUTPUT_DIR, "trash_classifier_model_v1.keras"))

def prepare_data():
    """
    Fetches and splits the TrashNet dataset.
    """
    print("Preparing data...")
    fetch_trash_zip()
    split_dataset()
    print("Data preparation complete.")


if __name__ == "__main__":
    prepare_data()