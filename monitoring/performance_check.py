import requests
import os
from pathlib import Path

API_URL = "http://localhost:8000/predict"

TEST_DATA_PATH = "data/processed/test"

correct = 0
total = 0

for label in ["Cat", "Dog"]:
    folder = Path(TEST_DATA_PATH) / label
    images = list(folder.glob("*.jpg"))[:5]   # take 5 samples per class

    for image_path in images:
        with open(image_path, "rb") as f:
            response = requests.post(API_URL, files={"file": f})

        prediction = response.json()["prediction"]

        if prediction == label:
            correct += 1

        total += 1

accuracy = correct / total if total > 0 else 0

print(f"Post-Deployment Accuracy on Sample: {accuracy:.2f}")