import os
import shutil
from src.data_preprocessing import create_processed_dirs, PROCESSED_DATA_PATH

def test_create_processed_dirs():
    if os.path.exists(PROCESSED_DATA_PATH):
        shutil.rmtree(PROCESSED_DATA_PATH)

    create_processed_dirs()

    assert os.path.exists(os.path.join(PROCESSED_DATA_PATH, "train", "Cat"))
    assert os.path.exists(os.path.join(PROCESSED_DATA_PATH, "train", "Dog"))
    assert os.path.exists(os.path.join(PROCESSED_DATA_PATH, "val", "Cat"))
    assert os.path.exists(os.path.join(PROCESSED_DATA_PATH, "test", "Dog"))