import io
import time
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI()

MODEL_PATH = "models/model.keras"
model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


@app.get("/health")
def health():
    return {"status": "Model is running"}

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_from_array(model, image_array):
    prediction = model.predict(image_array)[0][0]
    return float(prediction)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = get_model()
    start_time = time.time()

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    processed = preprocess_image(image)

    prediction = predict_from_array(model, processed)
    label = "Dog" if prediction > 0.5 else "Cat"

    latency = time.time() - start_time

    return {
        "prediction": label,
        "confidence": float(prediction),
        "latency_seconds": latency
    }