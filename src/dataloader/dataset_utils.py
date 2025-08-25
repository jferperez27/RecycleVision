import random
from pathlib import Path
import shutil

TRAIN_RATIO = 0.8
SEED = 42
MOVE = False
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
DATA_DIR = Path("data/raw/dataset-resized")
TRAIN_DIR = Path("data/processed/train")
TEST_DIR = Path("data/processed/test")

def split_dataset(data_dir=DATA_DIR, train_dir=TRAIN_DIR, test_dir=TEST_DIR, train_ratio=TRAIN_RATIO, seed=SEED, move=MOVE):
    """
    Splits dataset into training and testing sets for each class.
    """
    rng = random.seed(seed)

    for cls in CLASSES:
        class_dir = data_dir / cls
        images = list(class_dir.glob("*"))
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        train_class_dir = train_dir / cls
        test_class_dir = test_dir / cls
        train_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)
        for img in train_images:
            if move:
                shutil.move(str(img), train_class_dir / img.name)
            else:
                shutil.copy(str(img), train_class_dir / img.name)
        for img in test_images:
            if move:
                shutil.move(str(img), test_class_dir / img.name)
            else:
                shutil.copy(str(img), test_class_dir / img.name)
        print(f"Class '{cls}': {len(train_images)} train, {len(test_images)} test images.")
    print("Dataset split complete.")

if __name__ == "__main__":
    print("Splitting dataset into training and testing sets...")
    split_dataset()