import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

RAW_DATA_PATH = "data/raw/PetImages"
PROCESSED_DATA_PATH = "data/processed"
IMAGE_SIZE = (224, 224)

def create_processed_dirs():
    for split in ["train", "val", "test"]:
        for label in ["Cat", "Dog"]:
            os.makedirs(os.path.join(PROCESSED_DATA_PATH, split, label), exist_ok=True)

def remove_corrupted_images(path):
    for label in ["Cat", "Dog"]:
        folder = os.path.join(path, label)
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                img = Image.open(file_path)
                img.verify()
            except:
                print(f"Removing corrupted file: {file_path}")
                os.remove(file_path)

def split_data():
    for label in ["Cat", "Dog"]:
        images = list(Path(os.path.join(RAW_DATA_PATH, label)).glob("*.jpg"))
        train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

        for img in train_imgs:
            shutil.copy(img, os.path.join(PROCESSED_DATA_PATH, "train", label))

        for img in val_imgs:
            shutil.copy(img, os.path.join(PROCESSED_DATA_PATH, "val", label))

        for img in test_imgs:
            shutil.copy(img, os.path.join(PROCESSED_DATA_PATH, "test", label))

if __name__ == "__main__":
    remove_corrupted_images(RAW_DATA_PATH)
    create_processed_dirs()
    split_data()
    print("Data preprocessing completed.")