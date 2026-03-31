"""
app.py - Intelligent System Classification App
- ML Model  : Sports Ball Classification  (HOG + Random Forest)
- Neural Net: Sports Image Classification (MobileNetV2, 100 categories)
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# ── PATHS ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
BALL_DIR   = os.path.join(BASE_DIR, "ball",   "train")
SPORTS_DIR = os.path.join(BASE_DIR, "sports", "train")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

ML_IMG_SIZE = 64
NN_IMG_SIZE = 128

# Ball emoji map
BALL_EMOJI = {
    "american_football": "🏈", "baseball": "⚾", "basketball": "🏀",
    "billiard_ball": "🎱",     "bowling_ball": "🎳", "cricket_ball": "🏏",
    "football": "⚽",          "golf_ball": "⛳",    "hockey_ball": "🏑",
    "hockey_puck": "🏒",       "rugby_ball": "🏉",   "shuttlecock": "🏸",
}

BALL_DISPLAY = {
    "american_football": "American Football", "baseball": "Baseball",
    "basketball": "Basketball",               "billiard_ball": "Billiard Ball",
    "bowling_ball": "Bowling Ball",           "cricket_ball": "Cricket Ball",
    "football": "Football (Soccer)",          "golf_ball": "Golf Ball",
    "hockey_ball": "Hockey Ball",             "hockey_puck": "Hockey Puck",
    "rugby_ball": "Rugby Ball",               "shuttlecock": "Shuttlecock",
}

# ── CACHED DATA ───────────────────────────────────────────────

@st.cache_data
def load_model_info():
    p = os.path.join(MODEL_DIR, "model_info.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_history():
    p = os.path.join(MODEL_DIR, "history.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}

@st.cache_resource
def load_ml_model():
    import joblib
    mp = os.path.join(MODEL_DIR, "model_ml.joblib")
    lp = os.path.join(MODEL_DIR, "label_encoder_ml.joblib")
    if os.path.exists(mp) and os.path.exists(lp):
        return joblib.load(mp), joblib.load(lp)
    return None, None

@st.cache_resource
def load_nn_model():
    import tensorflow as tf
    for name in ["model_nn.keras", "model_nn_best.keras"]:
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p):
            return tf.keras.models.load_model(p)
    return None

# ── IMAGE UTILS ───────────────────────────────────────────────

def preprocess_ml(uploaded, size=ML_IMG_SIZE):
    img = Image.open(uploaded).convert("RGB").resize((size,size), Image.LANCZOS)
    return np.array(img)

def preprocess_nn(uploaded, size=NN_IMG_SIZE):
    img = Image.open(uploaded).convert("RGB").resize((size,size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def extract_hog(img):
    from skimage.feature import hog
    return hog(img, orientations=9, pixels_per_cell=(8,8),
               cells_per_block=(2,2), channel_axis=-1)

# ── CHART HELPERS ─────────────────────────────────────────────

def bar_chart(probs, labels, top_k=5, color="#6C63FF"):
    idx  = np.argsort(probs)[::-1][:top_k]
    vals = [probs[i]*100 for i in idx]
    labs = [BALL_DISPLAY.get(str(labels[i]), str(labels[i]).replace("_"," ").title())
            for i in idx]
    fig, ax = plt.subplots(figsize=(7, 3))
    cs      = [color if i==0 else "#DDD8F8" for i in range(len(labs))]
    ax.barh(labs[::-1], vals[::-1], color=cs[::-1], height=0.55)
    for i,(v,_) in enumerate(zip(vals[::-1], labs[::-1])):
        ax.text(v+0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0,105); ax.set_xlabel("ความน่าจะเป็น (%)")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

def training_history_chart(hist):
    if not hist:
        st.info("ไม่พบข้อมูล training history")
        return
    phases = []
    if "p1_acc" in hist:
        phases.append(("Phase 1 — Train Top Layers",
                        hist["p1_acc"], hist["p1_val_acc"],
                        hist["p1_loss"], hist["p1_val_loss"]))
    if "p2_acc" in hist:
        phases.append(("Phase 2 — Fine-Tuning",
                        hist["p2_acc"], hist["p2_val_acc"],
                        hist["p2_loss"], hist["p2_val_loss"]))
    for title, ta, va, tl, vl in phases:
        st.markdown(f"**{title}**")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        ep = range(1, len(ta)+1)
        axes[0].plot(ep, ta, label="Train", color="#6C63FF")
        axes[0].plot(ep, va, label="Val",   color="#FF6584", ls="--")
        axes[0].set_title("Accuracy"); axes[0].legend()
        axes[0].spines[["top","right"]].set_visible(False)
        axes[1].plot(ep, tl, label="Train", color="#6C63FF")
        axes[1].plot(ep, vl, label="Val",   color="#FF6584", ls="--")
        axes[1].set_title("Loss"); axes[1].legend()
        axes[1].spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

def sample_image(folder, label):
    """แสดงรูปตัวอย่างจาก dataset"""
    path = os.path.join(folder, label)
    if not os.path.isdir(path):
        return None
    files = [f for f in os.listdir(path)
             if f.lower().endswith((".jpg",".jpeg",".png"))]
    if files:
        return os.path.join(path, files[0])
    return None

def confidence_badge(conf):
    if conf >= 70:
        st.success(f"ความมั่นใจ: **{conf:.1f}%** ✅")
    elif conf >= 40:
        st.warning(f"ความมั่นใจ: **{conf:.1f}%** ⚠️")
    else:
        st.error(f"ความมั่นใจ: **{conf:.1f}%** ❌ (ต่ำ)")

# ════════════════════════════════════════════════════════════
# PAGE 1: อธิบาย ML Model (Ball)
# ════════════════════════════════════════════════════════════

def page_explain_ml(info):
    st.title("🎱 อธิบาย ML Model")
    st.caption("Dataset: Sports Ball Images | Task: จำแนกประเภทบอลกีฬา")
    st.markdown("---")

    # Dataset
    st.header("1. ข้อมูล Dataset")
    n_cls = info.get("ml_num_classes", 12)
    c1, c2, c3 = st.columns(3)
    c1.metric("ประเภทบอล", f"{n_cls} ประเภท")
    c2.metric("ขนาดรูปภาพ", f"{info.get('ml_img_size',64)}×{info.get('ml_img_size',64)} px")
    c3.metric("ML Accuracy", f"{info.get('ml_accuracy','?')}%")

    classes = info.get("ml_classes", list(BALL_DISPLAY.keys()))
    st.markdown("**ประเภทบอลทั้งหมด:**")
    cols = st.columns(4)
    for i, cls in enumerate(classes):
        emoji = BALL_EMOJI.get(cls, "🎯")
        name  = BALL_DISPLAY.get(cls, cls.replace("_"," ").title())
        cols[i%4].markdown(f"{emoji} {name}")

    # Data Prep
    st.header("2. การเตรียมข้อมูล & Data Augmentation")
    st.markdown("""
| Augmentation | รายละเอียด |
|---|---|
| Flip Horizontal | กลับภาพซ้าย-ขวา |
| Rotation +15° | หมุนตามเข็มนาฬิกา 15° |
| Rotation -15° | หมุนทวนเข็มนาฬิกา 15° |
| **รวม** | **4 รูปต่อภาพต้นฉบับ** |
    """)

    # HOG
    st.header("3. Feature Extraction — HOG")
    st.markdown("""
ใช้ **Histogram of Oriented Gradients (HOG)** จับรูปร่าง ขอบ และพื้นผิวของบอล:

| Parameter | ค่า |
|---|---|
| `orientations` | 9 |
| `pixels_per_cell` | (8, 8) |
| `cells_per_block` | (2, 2) |
| Feature vector | ~2,916 ค่าต่อรูป |

HOG ดีสำหรับบอลเพราะ:
- บอลแต่ละชนิดมี **รูปร่างและ texture เฉพาะตัว**
- ทนต่อการเปลี่ยนแสงและสี
    """)

    # Model
    st.header("4. Random Forest Classifier")
    st.markdown("""
```
HOG Features (2,916 ค่า)
        ↓
Random Forest
  ├── Tree 1  →  prediction
  ├── Tree 2  →  prediction
  ├── Tree 3  →  prediction
  └── Tree N  →  prediction
        ↓
  Majority Vote
        ↓
   Ball Class
```
- **150 Decision Trees** รวม votes กัน
- แต่ละ tree ใช้ random subset ของ features
- ป้องกัน overfitting ด้วย bagging
    """)

    # Accuracy
    st.header("5. ผลลัพธ์")
    st.metric("Accuracy (Test Set)", f"{info.get('ml_accuracy','?')}%")

    st.header("6. อ้างอิง")
    st.markdown("- [Sports Balls Dataset — Kaggle](https://www.kaggle.com/)")

# ════════════════════════════════════════════════════════════
# PAGE 2: อธิบาย Neural Network (Sports)
# ════════════════════════════════════════════════════════════

def page_explain_nn(info, hist):
    st.title("🧠 อธิบาย Neural Network")
    st.caption("Dataset: Sports Image Classification (100 ประเภทกีฬา) | Model: MobileNetV2")
    st.markdown("---")

    st.header("1. ข้อมูล Dataset")
    n_cls = info.get("nn_num_classes", 100)
    c1, c2, c3 = st.columns(3)
    c1.metric("ประเภทกีฬา", f"{n_cls} ประเภท")
    c2.metric("ขนาดรูปภาพ", f"{info.get('nn_img_size',128)}×{info.get('nn_img_size',128)} px")
    c3.metric("NN Accuracy", f"{info.get('nn_accuracy','?')}%")

    classes = info.get("nn_classes", [])
    if classes:
        st.markdown("**ตัวอย่างประเภทกีฬา (20 แรก):**")
        cols = st.columns(5)
        for i, cls in enumerate(classes[:20]):
            cols[i%5].markdown(f"- {cls.title()}")

    st.header("2. สถาปัตยกรรมโมเดล (MobileNetV2)")
    st.markdown("""
ใช้ **Transfer Learning** จาก MobileNetV2 (pre-trained ImageNet):

```
Input (128×128×3)
      ↓
MobileNetV2 Base   ← Pre-trained (ImageNet)
      ↓
GlobalAveragePooling2D
      ↓
BatchNormalization
      ↓
Dense(256, ReLU)
      ↓
Dropout(0.4)
      ↓
Dense(N_classes, Softmax)
```
MobileNetV2 ใช้ **Depthwise Separable Convolution** ทำให้เร็วและเบากว่า VGG/ResNet
    """)

    st.header("3. กลยุทธ์การ Train (2 Phase)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
#### Phase 1 — Top Layers
- **Freeze** MobileNetV2 ทั้งหมด
- Train เฉพาะ layers ใหม่
- LR = 1e-3, 5 epochs
        """)
    with c2:
        st.markdown("""
#### Phase 2 — Fine-Tuning
- **Unfreeze** 30 layers สุดท้าย
- Train ด้วย LR ต่ำ (1e-5)
- 10 epochs + EarlyStopping
        """)

    st.header("4. Training History")
    training_history_chart(hist)

    st.header("5. อ้างอิง")
    st.markdown("- [Sports Classification Dataset — Kaggle](https://www.kaggle.com/datasets/gpiosenka/sports-classification)")

# ════════════════════════════════════════════════════════════
# PAGE 3: ทดสอบ ML (Ball)
# ════════════════════════════════════════════════════════════

def page_test_ml():
    st.title("🎱 ทดสอบ ML Model — Sports Ball")
    st.caption("อัปโหลดรูปบอลกีฬา → โมเดลจะจำแนกประเภทบอล")
    st.markdown("---")

    model, le = load_ml_model()
    if model is None:
        st.error("❌ ยังไม่พบโมเดล ML — กรุณารัน: `python train_ml.py`")
        return
    st.success("✅ โหลดโมเดล ML สำเร็จ")

    # Sample images grid
    with st.expander("🖼️ ดูตัวอย่างรูปบอลแต่ละประเภท"):
        gcols = st.columns(6)
        for i, cls in enumerate(BALL_DISPLAY.keys()):
            sp = sample_image(BALL_DIR, cls)
            if sp:
                gcols[i%6].image(sp,
                    caption=f"{BALL_EMOJI.get(cls,'')} {BALL_DISPLAY.get(cls,cls)}",
                    width=90)

    st.markdown("### อัปโหลดรูปบอล")
    uploaded = st.file_uploader("เลือกรูปภาพ (JPG/PNG)",
                                type=["jpg","jpeg","png"], key="ml_up")

    if uploaded:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(uploaded, caption="รูปที่อัปโหลด", width=200)

        with st.spinner("🔍 กำลังวิเคราะห์..."):
            img  = preprocess_ml(uploaded)
            feat = extract_hog(img).reshape(1, -1)
            prob = model.predict_proba(feat)[0]
            idx  = int(np.argmax(prob))
            label = le.classes_[idx]

        emoji   = BALL_EMOJI.get(label, "🎯")
        display = BALL_DISPLAY.get(label, label.replace("_"," ").title())
        conf    = float(prob[idx]) * 100

        st.markdown("---")
        st.subheader("🎯 ผลการวิเคราะห์")

        with col2:
            st.markdown(f"# {emoji} {display}")
            confidence_badge(conf)

            # Sample from dataset
            sp = sample_image(BALL_DIR, label)
            if sp:
                st.image(sp, caption=f"ตัวอย่างจาก dataset: {display}", width=150)

        st.markdown("---")
        st.markdown("#### 📊 Top 5 ความน่าจะเป็น")
        bar_chart(prob, le.classes_, top_k=5, color="#6C63FF")

# ════════════════════════════════════════════════════════════
# PAGE 4: ทดสอบ Neural Network (Sports)
# ════════════════════════════════════════════════════════════

def page_test_nn(info):
    st.title("🧠 ทดสอบ Neural Network — Sports")
    st.caption("อัปโหลดรูปกีฬา → โมเดลจะจำแนกประเภทกีฬา")
    st.markdown("---")

    model = load_nn_model()
    if model is None:
        st.error("❌ ยังไม่พบโมเดล NN — กรุณารัน: `python train_nn.py`")
        return
    st.success("✅ โหลดโมเดล Neural Network สำเร็จ")

    classes = info.get("nn_classes", [])
    if not classes and os.path.isdir(SPORTS_DIR):
        classes = sorted([d for d in os.listdir(SPORTS_DIR)
                          if os.path.isdir(os.path.join(SPORTS_DIR, d))])

    st.markdown("### อัปโหลดรูปกีฬา")
    uploaded = st.file_uploader("เลือกรูปภาพ (JPG/PNG)",
                                type=["jpg","jpeg","png"], key="nn_up")

    if uploaded:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(uploaded, caption="รูปที่อัปโหลด", width=200)

        with st.spinner("🔍 กำลังวิเคราะห์..."):
            img_in = preprocess_nn(uploaded)
            prob   = model.predict(img_in, verbose=0)[0]
            idx    = int(np.argmax(prob))
            label  = classes[idx] if idx < len(classes) else f"class_{idx}"

        display = label.replace("_"," ").title()
        conf    = float(prob[idx]) * 100

        st.markdown("---")
        st.subheader("🎯 ผลการวิเคราะห์")

        with col2:
            st.markdown(f"# 🏅 {display}")
            confidence_badge(conf)

            sp = sample_image(SPORTS_DIR, label)
            if sp:
                st.image(sp, caption=f"ตัวอย่าง: {display}", width=150)

        st.markdown("---")
        st.markdown("#### 📊 Top 5 ความน่าจะเป็น")
        bar_chart(prob, classes, top_k=5, color="#FF6584")

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Intelligent System — Sports Classification",
        page_icon="🎱",
        layout="wide",
    )

    info = load_model_info()
    hist = load_history()

    with st.sidebar:
        st.markdown("# 🤖 Intelligent System")
        st.markdown("##### Sports Classification")
        st.markdown("---")

        page = st.radio("เลือกหน้า", [
            "🎱 อธิบาย ML Model",
            "🧠 อธิบาย Neural Network",
            "🎯 ทดสอบ ML (บอล)",
            "🏅 ทดสอบ NN (กีฬา)",
        ], label_visibility="collapsed")

        st.markdown("---")
        st.markdown(
            f"""
<div style="background:linear-gradient(135deg,#f0f4ff,#fff0f5);
padding:16px;border-radius:12px;font-size:13px;line-height:2">
<b>🎱 ML Dataset:</b><br>
&ensp;Sports Balls ({info.get('ml_num_classes',12)} ประเภท)<br>
<b>🎯 ML Accuracy:</b>
<span style="color:#6C63FF;font-weight:700">
&ensp;{info.get('ml_accuracy','?')}%</span><br><br>
<b>🏅 NN Dataset:</b><br>
&ensp;Sports ({info.get('nn_num_classes',100)} ประเภท)<br>
<b>🎯 NN Accuracy:</b>
<span style="color:#FF6584;font-weight:700">
&ensp;{info.get('nn_accuracy','?')}%</span>
</div>""",
            unsafe_allow_html=True,
        )

    if   "ML Model"         in page: page_explain_ml(info)
    elif "Neural Network"   in page: page_explain_nn(info, hist)
    elif "ML (บอล)"       in page:  page_test_ml()
    elif "NN (กีฬา)"      in page:  page_test_nn(info)

if __name__ == "__main__":
    main()
