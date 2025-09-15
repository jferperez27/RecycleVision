from pathlib import Path
import pytest
from src.dataloader import dataset_utils
from src.dataloader.fetch_trashnet_data import fetch_trash_zip

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def test_prepare_data():
    """
    Creates small dataset for testing purposes in tests/data directory.
    """
    data_src = Path("tests/data/raw/dataset-resized")
    train_src = Path("tests/data/processed/train")
    test_src = Path("tests/data/processed/test")

    if not data_src.exists() or not any(data_src.iterdir()):
        fetch_trash_zip(dir_path="tests/data/raw")

    dataset_utils.split_dataset(
        data_dir=data_src, 
        train_dir=train_src, 
        test_dir=test_src, 
        train_ratio=0.8, 
        seed=42, 
        move=False
        )
    
    assert train_src.exists() and test_src.exists()
    for cls in CLASS_NAMES:
        assert (train_src / cls).exists() and (test_src / cls).exists()
        assert len(list((train_src / cls).glob("*"))) > 0
        assert len(list((test_src / cls).glob("*"))) > 0 

def test_zip_fetch():
    """
    Tests if the dataset zip file is fetched and unzipped correctly.
    """
    data_dir = Path("tests/data/raw")
    fetch_trash_zip(dir_path=str(data_dir))
    unzipped_dir = data_dir / "dataset-resized"
    assert unzipped_dir.exists() and unzipped_dir.is_dir()
    for cls in CLASS_NAMES:
        assert (unzipped_dir / cls).exists() and (unzipped_dir / cls).is_dir()
        assert len(list((unzipped_dir / cls).glob("*"))) > 0
