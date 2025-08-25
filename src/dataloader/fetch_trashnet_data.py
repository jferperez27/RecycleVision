from huggingface_hub import hf_hub_download
from pathlib import Path
import zipfile


REPO = "garythung/trashnet"
FILE = "dataset-resized.zip"

def fetch_trash_zip(dir_path = "data/raw"):
    """
    Imports dataset zip file directly from HuggingFace.
    """
    dest = Path(dir_path)
    dest.mkdir(parents=True, exist_ok=True)

    zip_path = hf_hub_download(
        repo_id=REPO, 
        filename=FILE, 
        repo_type="dataset", 
        local_dir=dir_path,
        )
    print(f"Downloaded {zip_path}")
    root = unzip_data(dest, zip_path)

def unzip_data(dest: Path, zip_path: str):
    """
    Unzips the dataset zip file.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest)
    return dest / "dataset-resized"

if __name__ == "__main__":
    print("Fetching TrashNet dataset...")
    fetch_trash_zip()