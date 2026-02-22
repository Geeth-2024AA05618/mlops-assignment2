import os
from src.data_preprocessing import create_processed_dirs

def test_create_processed_dirs(tmp_path, monkeypatch):
    # Use temporary directory instead of real processed path
    monkeypatch.setattr(
        "src.data_preprocessing.PROCESSED_DATA_PATH",
        tmp_path
    )

    create_processed_dirs()

    assert (tmp_path / "train" / "Cat").exists()
    assert (tmp_path / "val" / "Dog").exists()