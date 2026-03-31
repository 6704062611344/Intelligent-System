"""
train_nn.py - Train Neural Network สำหรับ Sports Image Classification
Dataset: sports/ (train/test folders)
Model: EfficientNetB0 Transfer Learning (2-Phase)
"""

import os, json
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── CONFIG ──────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SPORTS_DIR = os.path.join(BASE_DIR, "sports")
CSV_PATH   = os.path.join(SPORTS_DIR, "sports.csv")
TRAIN_DIR  = os.path.join(SPORTS_DIR, "train")
TEST_DIR   = os.path.join(SPORTS_DIR, "test")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS_P1  = 10
EPOCHS_P2  = 20
LR_P1      = 1e-3
LR_P2      = 1e-5

PRETRAINED_H5 = os.path.join(SPORTS_DIR,
    "EfficientNetB0-100-(224 X 224)- 98.40.h5")

# ── MAIN ────────────────────────────────────────────────────

def get_classes_from_csv():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    return sorted(df["labels"].unique().tolist())

def get_classes_from_folder():
    if os.path.isdir(TRAIN_DIR):
        return sorted([d for d in os.listdir(TRAIN_DIR)
                       if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    return []

def save_info(acc, classes):
    info_path = os.path.join(MODEL_DIR, "model_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    info.update({
        "nn_accuracy":    acc,
        "nn_classes":     classes,
        "nn_num_classes": len(classes),
        "nn_img_size":    IMG_SIZE,
    })
    json.dump(info, open(info_path,"w"), indent=2)
    print(f"[SAVE] model_info.json updated  (accuracy={acc}%)")


def main():
    print("="*60)
    print("Sports Neural Network (EfficientNetB0)")
    print("="*60)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"[INFO] GPUs: {len(gpus)}")
    print(f"[INFO] TensorFlow: {tf.__version__}")

    # ── ลอง load โมเดลที่มีอยู่แล้ว ─────────────────────────
    if os.path.exists(PRETRAINED_H5):
        print(f"\n[INFO] พบโมเดล pre-trained: {os.path.basename(PRETRAINED_H5)}")

        # ลองหลายวิธี
        loaded = False
        for attempt, kwargs in enumerate([
            {"compile": False},
            {"compile": False, "safe_mode": False},
            {},
        ], 1):
            try:
                print(f"[TRY {attempt}] Load with {kwargs}...")
                model = tf.keras.models.load_model(PRETRAINED_H5, **kwargs)
                print("[OK] โหลดสำเร็จ!")

                # Save .keras
                out = os.path.join(MODEL_DIR, "model_nn.keras")
                model.save(out)
                print(f"[SAVE] model_nn.keras saved")

                classes = get_classes_from_csv() or get_classes_from_folder()
                save_info(98.40, classes)
                loaded = True
                break
            except Exception as e:
                print(f"  [FAIL] {type(e).__name__}: {str(e)[:120]}")

        if loaded:
            print("\nDONE! (ใช้โมเดลที่ train ไว้แล้ว 98.40%)")
            return

        print("\n[WARN] ไม่สามารถโหลดโมเดลเดิมได้ → จะ train ใหม่")

    # ── Train ใหม่ ────────────────────────────────────────────
    print("\n[INFO] เริ่ม train EfficientNetB0 ใหม่...")

    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=20,
        width_shift_range=0.15, height_shift_range=0.15,
        horizontal_flip=True, zoom_range=0.15,
        brightness_range=[0.7,1.3], fill_mode="nearest",
        validation_split=0.15,
    )
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

    train_ds = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical", subset="training")
    val_ds = val_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical", subset="validation")

    num_classes = train_ds.num_classes
    classes = sorted(train_ds.class_indices.keys())
    print(f"[INFO] Classes: {num_classes}")

    # Build Model
    base = EfficientNetB0(input_shape=(IMG_SIZE,IMG_SIZE,3),
                          include_top=False, weights="imagenet")
    base.trainable = False

    inp = tf.keras.Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out)

    print("\n[PHASE 1] Train top layers...")
    model.compile(optimizer=optimizers.Adam(LR_P1),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    cb1 = [callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                    monitor="val_accuracy")]
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_P1,
                   callbacks=cb1, verbose=1)

    print("\n[PHASE 2] Fine-tuning...")
    base.trainable = True
    model.compile(optimizer=optimizers.Adam(LR_P2),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    cb2 = [
        callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                monitor="val_accuracy"),
        callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR,"model_nn_best.keras"),
            save_best_only=True, monitor="val_accuracy"),
    ]
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_P2,
                   callbacks=cb2, verbose=1)

    best_acc = round(max(h2.history["val_accuracy"])*100, 2)
    model.save(os.path.join(MODEL_DIR,"model_nn.keras"))
    print(f"[SAVE] model_nn.keras saved")

    # Save history
    hist = {
        "p1_acc":h1.history["accuracy"], "p1_val_acc":h1.history["val_accuracy"],
        "p1_loss":h1.history["loss"],    "p1_val_loss":h1.history["val_loss"],
        "p2_acc":h2.history["accuracy"], "p2_val_acc":h2.history["val_accuracy"],
        "p2_loss":h2.history["loss"],    "p2_val_loss":h2.history["val_loss"],
    }
    json.dump(hist, open(os.path.join(MODEL_DIR,"history.json"),"w"), indent=2)

    save_info(best_acc, classes)
    print(f"\nDONE! Accuracy: {best_acc}%")


if __name__ == "__main__":
    main()
