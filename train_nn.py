"""
train_nn.py - Train Neural Network (FAST VERSION)
Dataset: sports/train/ (100 ประเภทกีฬา)
Model: MobileNetV2 @ 128x128  → ~30-60 นาที บน CPU
"""

import os, json
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── CONFIG ──────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SPORTS_DIR = os.path.join(BASE_DIR, "sports")
CSV_PATH   = os.path.join(SPORTS_DIR, "sports.csv")
TRAIN_DIR  = os.path.join(SPORTS_DIR, "train")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE    = 128   # เล็กลง 224→128 เร็วขึ้น ~3x
BATCH_SIZE  = 64    # ใหญ่ขึ้น = เร็วขึ้น
EPOCHS_P1   = 5     # Phase 1 รวดเร็ว
EPOCHS_P2   = 10    # Phase 2 fine-tune
LR_P1       = 1e-3
LR_P2       = 1e-5

# ── MAIN ────────────────────────────────────────────────────

def main():
    print("="*60)
    print("Sports NN Training  (MobileNetV2 @ 128×128 — FAST MODE)")
    print("="*60)
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {len(tf.config.list_physical_devices('GPU'))}")

    # ── Data Generators ───────────────────────────────────────
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode="nearest",
        validation_split=0.15,
    )
    val_gen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

    print("\n[INFO] Loading dataset...")
    train_ds = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )
    val_ds = val_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    num_classes = train_ds.num_classes
    classes = sorted(train_ds.class_indices.keys())
    print(f"[INFO] Classes: {num_classes} | Train: {train_ds.samples} | Val: {val_ds.samples}")

    # ── Build Model ───────────────────────────────────────────
    print("\n[INFO] Building MobileNetV2...")
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(256, activation="relu")(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out)

    total_params = model.count_params()
    print(f"[INFO] Parameters: {total_params:,}")

    # ── Phase 1: Train top layers ─────────────────────────────
    print(f"\n[PHASE 1] Train top layers ({EPOCHS_P1} epochs max)...")
    model.compile(
        optimizer=optimizers.Adam(LR_P1),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    cb1 = [
        callbacks.EarlyStopping(patience=3, restore_best_weights=True,
                                monitor="val_accuracy", verbose=1),
    ]
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_P1,
        callbacks=cb1,
        verbose=1,
    )
    best_p1 = max(h1.history["val_accuracy"])
    print(f"[PHASE 1] Best val accuracy: {best_p1*100:.2f}%")

    # ── Phase 2: Fine-tune ────────────────────────────────────
    print(f"\n[PHASE 2] Fine-tuning ({EPOCHS_P2} epochs max)...")
    # Unfreeze only top 30 layers of MobileNetV2
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(LR_P2),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    cb2 = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                monitor="val_accuracy", verbose=1),
        callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "model_nn_best.keras"),
            save_best_only=True, monitor="val_accuracy", verbose=0,
        ),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3,
                                    monitor="val_accuracy", verbose=1),
    ]
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_P2,
        callbacks=cb2,
        verbose=1,
    )
    best_p2 = max(h2.history["val_accuracy"])
    print(f"[PHASE 2] Best val accuracy: {best_p2*100:.2f}%")

    best_acc = round(max(best_p1, best_p2) * 100, 2)

    # ── Save Model ────────────────────────────────────────────
    save_path = os.path.join(MODEL_DIR, "model_nn.keras")
    model.save(save_path)
    print(f"\n[SAVE] model_nn.keras saved → {save_path}")

    # ── Save History ──────────────────────────────────────────
    hist = {
        "p1_acc":     h1.history["accuracy"],
        "p1_val_acc": h1.history["val_accuracy"],
        "p1_loss":    h1.history["loss"],
        "p1_val_loss":h1.history["val_loss"],
        "p2_acc":     h2.history["accuracy"],
        "p2_val_acc": h2.history["val_accuracy"],
        "p2_loss":    h2.history["loss"],
        "p2_val_loss":h2.history["val_loss"],
    }
    json.dump(hist, open(os.path.join(MODEL_DIR, "history.json"), "w"), indent=2)

    # ── Save Info ─────────────────────────────────────────────
    info_path = os.path.join(MODEL_DIR, "model_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    info.update({
        "nn_accuracy":    best_acc,
        "nn_classes":     classes,
        "nn_num_classes": num_classes,
        "nn_img_size":    IMG_SIZE,
        "nn_model":       "MobileNetV2",
    })
    json.dump(info, open(info_path, "w"), indent=2)
    print(f"[SAVE] model_info.json updated")

    print("\n" + "="*60)
    print(f"DONE!  NN Accuracy: {best_acc}%")
    print("="*60)


if __name__ == "__main__":
    main()
