"""
train_ml.py - Train ML Ensemble Model สำหรับ Pokemon Identification
Dataset: pokemon/ (809 ตัว, 1 รูปต่อ Pokemon)
Model: HOG features + Random Forest + SVM + XGBoost (Soft Voting)
"""

import os, json
import numpy as np
import pandas as pd
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
from PIL import Image

# ── CONFIG ──────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
POKE_DIR   = os.path.join(BASE_DIR, "pokemon")
IMAGE_DIR  = os.path.join(POKE_DIR, "images")
CSV_PATH   = os.path.join(POKE_DIR, "pokemon.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE   = 96
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8,8),
                  cells_per_block=(2,2), channel_axis=-1)

# ── HELPERS ─────────────────────────────────────────────────

def load_rgb(path, size=IMG_SIZE):
    try:
        img = Image.open(path).convert("RGBA")
        bg  = Image.new("RGBA", img.size, (255,255,255,255))
        bg.paste(img, mask=img.split()[3])
        return np.array(bg.convert("RGB").resize((size,size), Image.LANCZOS))
    except:
        return None

def augment(img):
    h, w = img.shape[:2]
    imgs = [img, cv2.flip(img, 1)]
    for ang in [-15, 15]:
        M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
        imgs.append(cv2.warpAffine(img, M, (w,h), borderValue=(255,255,255)))
    for g in [0.75, 1.25]:
        t = np.array([(i/255)**( 1/g)*255 for i in range(256)], np.uint8)
        imgs.append(cv2.LUT(img, t))
    return imgs   # 6 รูปต่อ Pokemon

def extract_hog(img):
    return hog(img, **HOG_PARAMS)

# ── MAIN ────────────────────────────────────────────────────

def main():
    print("="*60)
    print("Pokemon ML Training (HOG + Ensemble)")
    print("="*60)

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    X_list, y_list, loaded = [], [], []
    for _, row in df.iterrows():
        name = str(row["Name"]).strip().lower()
        path = os.path.join(IMAGE_DIR, f"{name}.png")
        if not os.path.exists(path):
            continue
        img = load_rgb(path)
        if img is None:
            continue
        for aug in augment(img):
            X_list.append(extract_hog(aug))
            y_list.append(name)
        loaded.append(name)

    print(f"[INFO] {len(loaded)} Pokemon | {len(X_list)} samples (after augmentation)")

    X  = np.array(X_list, dtype=np.float32)
    le = LabelEncoder()
    y  = le.fit_transform(np.array(y_list))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"[INFO] Train={len(X_train)} | Test={len(X_test)}")

    print("\n[1/3] Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"  Accuracy: {rf_acc*100:.2f}%")

    print("[2/3] SVM...")
    svm = SVC(kernel="rbf", C=10, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    print(f"  Accuracy: {svm_acc*100:.2f}%")

    print("[3/3] XGBoost...")
    xgb_m = xgb.XGBClassifier(n_estimators=100, max_depth=4,
                               learning_rate=0.1, eval_metric="mlogloss",
                               random_state=42, n_jobs=-1)
    xgb_m.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb_m.predict(X_test))
    print(f"  Accuracy: {xgb_acc*100:.2f}%")

    print("[Ensemble] Soft Voting...")
    ens = VotingClassifier([("rf",rf),("svm",svm),("xgb",xgb_m)],
                           voting="soft", n_jobs=-1)
    ens.fit(X_train, y_train)
    ens_acc = accuracy_score(y_test, ens.predict(X_test))
    print(f"  Accuracy: {ens_acc*100:.2f}%")

    joblib.dump(ens, os.path.join(MODEL_DIR, "model_ml.joblib"))
    joblib.dump(le,  os.path.join(MODEL_DIR, "label_encoder_ml.joblib"))

    # update model_info.json
    info_path = os.path.join(MODEL_DIR, "model_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    info.update({
        "ml_accuracy": round(ens_acc*100, 2),
        "ml_rf_acc":   round(rf_acc*100, 2),
        "ml_svm_acc":  round(svm_acc*100, 2),
        "ml_xgb_acc":  round(xgb_acc*100, 2),
        "ml_classes":  list(le.classes_),
        "ml_num_classes": len(le.classes_),
        "ml_img_size": IMG_SIZE,
    })
    json.dump(info, open(info_path,"w"), indent=2)

    print(f"\nDONE! Ensemble Accuracy: {ens_acc*100:.2f}%")

if __name__ == "__main__":
    main()
