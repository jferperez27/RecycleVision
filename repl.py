import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))  

import os
from src.config import *
import src.config as config
from src.dataloader import fetch_trashnet_data, dataset_utils
from src.train import train
from src.inference import load_model, custom_model_test_inference, model_inference

def have_data_dirs():
    try:
        os.listdir(DATA_DIR)
        os.listdir(TRAIN_DIR)
        os.listdir(TEST_DIR)
        return True
    except FileNotFoundError:
        return False
    
def prepare_data():
    fetch_trashnet_data.fetch_trash_zip()
    dataset_utils.split_dataset()

def run_tests():
    import pytest
    pytest.main(["-v", "tests/"])

def train_new_model():
    print("Set up training parameters (NOTE: blanks will use defaults from config.py):")
    batch_size = input(f"Batch size (default {BATCH_SIZE}): ").strip()
    epochs = input(f"Epochs (default {EPOCHS}): ").strip()
    file_name = input(f"Model file name (default {MODEL_NAME}): ").strip()
    if batch_size.isdigit():
        config.BATCH_SIZE = int(batch_size)
    if epochs.isdigit():
        config.EPOCHS = int(epochs)
    if file_name:
        if not file_name.endswith(".keras"):
            file_name += ".keras"
        config.MODEL_NAME = file_name
    print(f"Training model with batch size {config.BATCH_SIZE}, epochs {config.EPOCHS}, saving as {config.MODEL_NAME}")
    train()

def get_model():
    while True:
        if MEMORY["model"] is not None:
            load = input("Model already loaded. Would you like to load a different model? (y/n)")
            if load.lower() == 'n':
                print("Using previously loaded model.")
                return MEMORY["model"]
            elif load.lower() == 'y':
                break
        else:
            break
    print("Leave blank to use default model path 'models/trash_classifier_model_v3.keras'")
    model_path = input("Enter the path to the model file: ").strip()
    if not model_path:
        model_path = "models/trash_classifier_model_v3.keras"
    if not Path(model_path).exists():
        print(f"Model file '{model_path}' does not exist.")
        return None
    model = load_model(model_path)
    MEMORY["model"] = model
    print(f"Model loaded from '{model_path}'")
    return model

def run_test_images():
    if MEMORY["model"] is None:
        print("No model loaded. Please load a model first using the 'load_model' command.")
        return
    else:
        while True:
            custom_model_test_inference(MEMORY["model"])
            again = input("Test another image? (y/n): ").strip().lower()
            if again == 'n':
                break

def run_inference():
    if MEMORY["model"] is None:
        print("No model loaded. Please load a model first using the 'load_model' command.")
        return
    else:
        while True:
            image_path = input("Enter the path to the image file for inference (or 'exit' to return): ").strip()
            if image_path == '':
                print("Image path cannot be blank.")
                return
            if image_path.lower() == 'exit':
                break
            if not Path(image_path).exists():
                print(f"Image file '{image_path}' does not exist.")
                continue
            model_inference(image_path)

REPL_COMMANDS = {
    "test": run_tests,
    "train": train_new_model,
    "load_model": get_model,
    "test_model": run_test_images,
    "inference": run_inference,
}

MEMORY = {
    "model": None
}

def main():
    ## Check if data is imported and prepared
    if not have_data_dirs():
        while True:
            cmd = input("Data directories not found. Would you like to import and prepare the data now? (y/n)").strip().lower()
            if cmd == 'y':
                prepare_data()
                break
            elif cmd == 'n':
                print("Data directories are required to proceed. Exiting REPL.")
                return

    ## Main REPL loop
    print("RecycleVision REPL environment. Type 'help' for commands. Type 'exit' to quit.")
    while True:
        cmd = input("rv> ").strip().lower()

        if cmd == "exit":
            print("Exiting RecycleVision REPL. Goodbye!")
            break
        elif cmd == "help":
            print("Available commands:")
            print("  help - Show this help message")
            print("  test - Run tests")
            print("  train - Train a new model")
            print("  load_model - Load a trained model")
            print("  test_model - Test model predictions with test images")
            print("  inference - Run custom single image prediction")
            print("  exit - Exit the REPL")
        elif cmd in REPL_COMMANDS:
            REPL_COMMANDS[cmd]()
        else:
            print(f"Unknown command: '{cmd}'. Type 'help' for a list of commands.")
        print("RecycleVision REPL environment. Type 'help' for commands. Type 'exit' to quit.")

    pass

if __name__ == "__main__":
    main()