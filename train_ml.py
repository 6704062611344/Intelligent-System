"""
train_ml.py - Train ML Model สำหรับ Sports Ball Classification
Dataset: ball/train/ (12 ประเภทบอล)
Model: HOG features + Random Forest
"""

import os, json
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import cv2
from PIL import Image

# ── CONFIG ──────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
BALL_DIR  = os.path.join(BASE_DIR, "ball", "train")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE   = 64
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8,8),
                  cells_per_block=(2,2), channel_axis=-1)

# ── HELPERS ──────────────────────────────────────────────────

def load_rgb(path, size=IMG_SIZE):
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img.resize((size, size), Image.LANCZOS))
    except:
        return None

def augment(img):
    """4 รูปต่อภาพ: original + flip + rotate ±15°"""
    h, w = img.shape[:2]
    imgs = [img, cv2.flip(img, 1)]
    for ang in [-15, 15]:
        M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
        imgs.append(cv2.warpAffine(img, M, (w,h),
                                   borderValue=(255,255,255)))
    return imgs

def extract_hog(img):
    return hog(img, **HOG_PARAMS)

# ── MAIN ─────────────────────────────────────────────────────

def main():
    print("="*60)
    print("Sports Ball ML Training  (HOG + Random Forest)")
    print("="*60)

    # ── Load Dataset ─────────────────────────────────────────
    classes = sorted([d for d in os.listdir(BALL_DIR)
                      if os.path.isdir(os.path.join(BALL_DIR, d))])
    print(f"[INFO] Classes ({len(classes)}): {classes}")

    X_list, y_list = [], []
    for cls in classes:
        cls_dir = os.path.join(BALL_DIR, cls)
        imgs_files = [f for f in os.listdir(cls_dir)
                      if f.lower().endswith((".jpg",".jpeg",".png"))]
        for fname in imgs_files:
            img = load_rgb(os.path.join(cls_dir, fname))
            if img is None:
                continue
            for aug in augment(img):
                X_list.append(extract_hog(aug))
                y_list.append(cls)
        print(f"  {cls}: {len(imgs_files)} images → {len(imgs_files)*4} samples")

    print(f"\n[INFO] Total: {len(X_list)} samples | {len(classes)} classes")

    X  = np.array(X_list, dtype=np.float32)
    le = LabelEncoder()
    y  = le.fit_transform(np.array(y_list))

    print(f"[INFO] Feature size: {X.shape[1]} | RAM: {X.nbytes/1024**2:.1f} MB")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[INFO] Train={len(X_train)} | Test={len(X_test)}")

    # ── Train ─────────────────────────────────────────────────
    print("\n[Training] Random Forest (150 trees)...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    rf.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[Result] Accuracy: {acc*100:.2f}%")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=le.classes_))

    # ── Save ──────────────────────────────────────────────────
    ml_path = os.path.join(MODEL_DIR, "model_ml.joblib")
    joblib.dump(rf, ml_path, compress=3)
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder_ml.joblib"), compress=3)

    size_mb = os.path.getsize(ml_path) / 1024**2
    print(f"\n[SAVE] model_ml.joblib → {size_mb:.1f} MB")

    info_path = os.path.join(MODEL_DIR, "model_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    info.update({
        "ml_accuracy":    round(acc*100, 2),
        "ml_classes":     list(le.classes_),
        "ml_num_classes": len(le.classes_),
        "ml_img_size":    IMG_SIZE,
        "ml_model":       "Random Forest (HOG)",
        "ml_dataset":     "Sports Balls",
    })
    json.dump(info, open(info_path, "w"), indent=2)

    print("\n" + "="*60)
    print(f"DONE!  Accuracy: {acc*100:.2f}%  |  Model: {size_mb:.1f} MB")
    print("="*60)

if __name__ == "__main__":
    main()
