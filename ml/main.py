import sys
import os
import argparse
import torch

# ─────────────────────────────────────────────
# FIX PROJECT ROOT PATH (IMPORTANT)
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from ml.dataset import get_dataloaders
from models.cnn1 import CNN1
from ml.train import train
from ml.evaluate import evaluate


# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, choices=["pytorch", "tensorflow"], required=True)
    parser.add_argument("--firstname", type=str, default="Julianna")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    return parser.parse_args()


# ─────────────────────────────────────────────
# PYTORCH TRAINING PIPELINE
# ─────────────────────────────────────────────
def run_pytorch(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PyTorch] Training on device: {device}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # Model
    model = CNN1(num_classes=6)

    # Save path (safe cross-platform)
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    save_path = os.path.join(models_dir, f"{args.firstname}_model.pth")

    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=1e-4,
        device=device,
        save_path=save_path
    )

    # Load best model
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("[PyTorch] Evaluating best model...")
    evaluate(model, test_loader, device)


# ─────────────────────────────────────────────
# TENSORFLOW TRAINING PIPELINE
# ─────────────────────────────────────────────
def run_tensorflow(args):

    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks

    # Dataset paths (safe)
    train_dir = os.path.join(PROJECT_ROOT, "archive", "seg_train", "seg_train")
    test_dir  = os.path.join(PROJECT_ROOT, "archive", "seg_test", "seg_test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(228, 228),
        batch_size=args.batch_size,
        label_mode="int"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(228, 228),
        batch_size=args.batch_size,
        label_mode="int"
    )

    # Normalize
    norm = layers.Rescaling(1. / 255)
    train_ds = train_ds.map(lambda x, y: (norm(x), y))
    val_ds = val_ds.map(lambda x, y: (norm(x), y))

    # Model
    model = models.Sequential([
        layers.Input(shape=(228, 228, 3)),

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(6, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # Save path
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    save_path = os.path.join(models_dir, f"{args.firstname}_model.keras")

    # Callbacks
    cb_list = [
        callbacks.ModelCheckpoint(
            save_path,
            save_best_only=True,
            monitor="val_loss"
        ),
        callbacks.EarlyStopping(
            patience=5,
            monitor="val_loss",
            restore_best_weights=True
        )
    ]

    print(f"[TensorFlow] Training started → {save_path}")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cb_list
    )

    print(f"[TensorFlow] Model saved → {save_path}")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    if args.model == "pytorch":
        run_pytorch(args)

    elif args.model == "tensorflow":
        run_tensorflow(args)