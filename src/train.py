import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf
tf.data.experimental.enable_debug_mode()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR = "data/processed"
MODEL_DIR = "models"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 2

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Cats_vs_Dogs")

def load_data():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Optional safety mapping (not mandatory but safe)
    def safe_map(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label

    train_ds = train_ds.map(safe_map, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(safe_map, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(safe_map, num_parallel_calls=tf.data.AUTOTUNE)

    # 🔥 ADD IGNORE_ERRORS HERE
    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
    val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())

    # Prefetch for performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

def plot_and_log(history):
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])
    plt.savefig("accuracy.png")
    mlflow.log_artifact("accuracy.png")

def main():
    train_ds, val_ds, test_ds = load_data()
    model = build_model()

    with mlflow.start_run():
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS
        )

        test_loss, test_acc = model.evaluate(test_ds)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)

        # Confusion Matrix
        y_true = []
        y_pred = []

        for images, labels in test_ds:
            preds = model.predict(images)
            y_true.extend(labels.numpy())
            y_pred.extend((preds > 0.5).astype(int).flatten())

        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, "model.keras")
        model.save(model_path)

        mlflow.tensorflow.log_model(model, "model")

if __name__ == "__main__":
    main()