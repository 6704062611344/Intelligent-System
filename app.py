"""
app.py - Intelligent System Classification App
- ML Model: Pokemon Identification (HOG + RF/SVM/XGBoost Ensemble)
- Neural Network: Sports Image Classification (EfficientNetB0)
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
POKE_DIR   = os.path.join(BASE_DIR, "pokemon")
SPORTS_DIR = os.path.join(BASE_DIR, "sports")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

POKE_CSV    = os.path.join(POKE_DIR,   "pokemon.csv")
POKE_IMGS   = os.path.join(POKE_DIR,   "images")
SPORTS_CSV  = os.path.join(SPORTS_DIR, "sports.csv")
SPORTS_TRAIN= os.path.join(SPORTS_DIR, "train")

ML_IMG_SIZE = 96
NN_IMG_SIZE = 224

# Pokemon type colours
TYPE_COLORS = {
    "Fire":"#F08030","Water":"#6890F0","Grass":"#78C850","Electric":"#F8D030",
    "Ice":"#98D8D8","Fighting":"#C03028","Poison":"#A040A0","Ground":"#E0C068",
    "Flying":"#A890F0","Psychic":"#F85888","Bug":"#A8B820","Rock":"#B8A038",
    "Ghost":"#705898","Dragon":"#7038F8","Dark":"#705848","Steel":"#B8B8D0",
    "Fairy":"#EE99AC","Normal":"#A8A878",
}

# ── CACHED DATA ───────────────────────────────────────────────

@st.cache_data
def load_pokemon_df():
    df = pd.read_csv(POKE_CSV)
    df.columns = [c.strip() for c in df.columns]
    df["Name"] = df["Name"].str.strip().str.lower()
    return df

@st.cache_data
def load_sports_df():
    df = pd.read_csv(SPORTS_CSV)
    df.columns = [c.strip() for c in df.columns]
    return df

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
    # ลอง load จาก .h5 โดยตรง
    h5 = os.path.join(SPORTS_DIR, "EfficientNetB0-100-(224 X 224)- 98.40.h5")
    if os.path.exists(h5):
        return tf.keras.models.load_model(h5)
    return None

# ── IMAGE UTILS ───────────────────────────────────────────────

def preprocess_for_ml(uploaded, size=ML_IMG_SIZE):
    img = Image.open(uploaded).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255,255,255,255))
    bg.paste(img, mask=img.split()[3])
    img = bg.convert("RGB").resize((size,size), Image.LANCZOS)
    return np.array(img)

def preprocess_for_nn(uploaded, size=NN_IMG_SIZE):
    img = Image.open(uploaded).convert("RGB").resize((size,size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def extract_hog(img):
    from skimage.feature import hog
    return hog(img, orientations=9, pixels_per_cell=(8,8),
               cells_per_block=(2,2), channel_axis=-1)

# ── LOOKUP ────────────────────────────────────────────────────

def lookup_pokemon(name, df):
    row = df[df["Name"] == name.lower()]
    if row.empty:
        return {"name":name.title(),"type1":"","type2":"","evolution":""}
    r = row.iloc[0]
    def safe(v): return str(v).strip() if pd.notna(v) and str(v).strip() not in ("nan","") else ""
    return {"name":str(r["Name"]).title(), "type1":safe(r["Type1"]),
            "type2":safe(r["Type2"]), "evolution":safe(r["Evolution"])}

# ── UI HELPERS ────────────────────────────────────────────────

def type_badge(t):
    c = TYPE_COLORS.get(t,"#888")
    st.markdown(f'<span style="background:{c};color:white;padding:4px 12px;'
                f'border-radius:12px;font-weight:600;font-size:13px;margin:2px">'
                f'{t}</span>', unsafe_allow_html=True)

def bar_chart(probs, labels, top_k=5, color="#6C63FF"):
    idx  = np.argsort(probs)[::-1][:top_k]
    vals = [probs[i]*100 for i in idx]
    labs = [str(labels[i]).title() for i in idx]
    fig, ax = plt.subplots(figsize=(7,3))
    colors = [color if i==0 else "#D8D4F5" for i in range(len(labs))]
    ax.barh(labs[::-1], vals[::-1], color=colors[::-1], height=0.55)
    for i,(v,l) in enumerate(zip(vals[::-1],labs[::-1])):
        ax.text(v+0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0,100); ax.set_xlabel("ความน่าจะเป็น (%)")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

def history_chart(hist):
    if not hist:
        st.info("ไม่พบข้อมูล training history")
        return
    phases = []
    if "p1_acc" in hist:
        phases.append(("Phase 1 — Train Top Layers",
                        hist["p1_acc"],hist["p1_val_acc"],
                        hist["p1_loss"],hist["p1_val_loss"]))
    if "p2_acc" in hist:
        phases.append(("Phase 2 — Fine-Tuning",
                        hist["p2_acc"],hist["p2_val_acc"],
                        hist["p2_loss"],hist["p2_val_loss"]))
    for title, ta,va,tl,vl in phases:
        st.markdown(f"**{title}**")
        fig, axes = plt.subplots(1,2,figsize=(10,3))
        ep = range(1, len(ta)+1)
        axes[0].plot(ep,ta,label="Train",color="#6C63FF")
        axes[0].plot(ep,va,label="Val",  color="#FF6584",ls="--")
        axes[0].set_title("Accuracy"); axes[0].legend()
        axes[0].spines[["top","right"]].set_visible(False)
        axes[1].plot(ep,tl,label="Train",color="#6C63FF")
        axes[1].plot(ep,vl,label="Val",  color="#FF6584",ls="--")
        axes[1].set_title("Loss"); axes[1].legend()
        axes[1].spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ════════════════════════════════════════════════════════════
# PAGE 1: อธิบาย ML Model (Pokemon)
# ════════════════════════════════════════════════════════════

def page_explain_ml(info):
    st.title("🤖 อธิบาย ML Model")
    st.caption("Dataset: Pokemon Images | Task: ระบุชื่อ Pokemon จากรูปภาพ")
    st.markdown("---")

    # Dataset overview
    st.header("1. ข้อมูล Dataset")
    c1,c2,c3 = st.columns(3)
    c1.metric("จำนวน Pokemon", f"{info.get('ml_num_classes',809)} ตัว")
    c2.metric("ขนาดรูปภาพ", f"{info.get('ml_img_size',96)}×{info.get('ml_img_size',96)} px")
    c3.metric("ข้อมูลเพิ่มเติม", "Type1, Type2, Evolution")
    st.markdown("""
**Source:** Pokemon Images & Stats dataset — รูป PNG ของ Pokemon 809 ตัว (1 รูปต่อตัว)

| คอลัมน์ | คำอธิบาย |
|---------|---------|
| Name | ชื่อ Pokemon |
| Type1 | ประเภทหลัก (เช่น Fire, Water) |
| Type2 | ประเภทรอง (ถ้ามี) |
| Evolution | Pokemon ที่ evolve ต่อ (ถ้ามี) |
    """)

    # Data Prep
    st.header("2. การเตรียมข้อมูล & Data Augmentation")
    st.markdown("""
เนื่องจากมีรูปแค่ **1 รูปต่อ Pokemon** จึงใช้ Data Augmentation เพิ่มข้อมูล:

| Augmentation | รายละเอียด |
|---|---|
| Flip Horizontal | กลับภาพซ้าย-ขวา |
| Rotation | หมุน ±15° |
| Brightness | ปรับความสว่าง (γ=0.75, 1.25) |
| **รวม** | **6 รูปต่อ Pokemon (~4,800+ samples)** |
    """)

    # HOG
    st.header("3. Feature Extraction — HOG")
    st.markdown("""
ใช้ **Histogram of Oriented Gradients (HOG)** จับรูปร่างและขอบ Pokemon:
- `orientations = 9`
- `pixels_per_cell = (8, 8)`
- `cells_per_block = (2, 2)`
- Feature vector: **~8,100 ค่าต่อรูป**
    """)

    # Models
    st.header("4. ทฤษฎีของแต่ละโมเดล")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("#### 🌲 Random Forest")
        st.markdown("- สร้าง Decision Tree 200 ต้น\n- ลด overfitting ด้วย random sampling\n- robust ต่อ noise")
    with c2:
        st.markdown("#### 📐 SVM")
        st.markdown("- หา hyperplane แบ่ง class\n- RBF kernel (จัดการ non-linear)\n- C=10, probability=True")
    with c3:
        st.markdown("#### ⚡ XGBoost")
        st.markdown("- Gradient Boosting Tree\n- 200 estimators, depth=6\n- learning_rate=0.1")

    st.markdown("""
รวม 3 โมเดลด้วย **Soft Voting** — เฉลี่ย probability แล้วเลือก class ที่มีค่าสูงสุด
    """)

    # Accuracy Table
    st.header("5. ผลลัพธ์ (Accuracy)")
    acc_df = pd.DataFrame({
        "โมเดล": ["Random Forest","SVM","XGBoost","**Ensemble (Soft Voting)**"],
        "Accuracy": [
            f"~{info.get('ml_rf_acc','?')}%",
            f"~{info.get('ml_svm_acc','?')}%",
            f"~{info.get('ml_xgb_acc','?')}%",
            f"**{info.get('ml_accuracy','?')}%**",
        ]
    })
    st.table(acc_df)

    # Ref
    st.header("6. อ้างอิง")
    st.markdown("- [Pokemon Images & Types — Kaggle](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types)")

# ════════════════════════════════════════════════════════════
# PAGE 2: อธิบาย Neural Network (Sports)
# ════════════════════════════════════════════════════════════

def page_explain_nn(info, hist):
    st.title("🧠 อธิบาย Neural Network")
    st.caption("Dataset: Sports Image Classification (100 ประเภทกีฬา) | Model: EfficientNetB0")
    st.markdown("---")

    # Dataset
    st.header("1. ข้อมูล Dataset")
    n_cls = info.get("nn_num_classes", 100)
    c1,c2,c3 = st.columns(3)
    c1.metric("ประเภทกีฬา", f"{n_cls} ประเภท")
    c2.metric("ขนาดรูปภาพ", f"{info.get('nn_img_size',224)}×{info.get('nn_img_size',224)} px")
    c3.metric("NN Accuracy", f"{info.get('nn_accuracy','?')}%")
    st.markdown("""
**Source:** Sports Image Classification dataset — รูปภาพกีฬากว่า 100 ประเภท
พร้อม label ชื่อกีฬา ใช้สำหรับ classification
    """)

    # Classes grid
    classes = info.get("nn_classes", [])
    if classes:
        st.markdown("**ตัวอย่างประเภทกีฬาบางส่วน:**")
        cols = st.columns(5)
        for i, cls in enumerate(classes[:20]):
            cols[i%5].markdown(f"- {cls.title()}")

    # Architecture
    st.header("2. สถาปัตยกรรมโมเดล (EfficientNetB0)")
    st.markdown("""
ใช้ **Transfer Learning** จาก EfficientNetB0 (pre-trained ImageNet):

```
Input (224×224×3)
    ↓
EfficientNetB0 Base  ← Pre-trained weights (ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(256, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(N_classes, Softmax)
```
    """)

    # Training Strategy
    st.header("3. กลยุทธ์การ Train (2 Phase)")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
#### Phase 1 — Train Top Layers
- **Freeze** EfficientNetB0 ทั้งหมด
- Train เฉพาะ layers ที่เพิ่มใหม่
- LR = 1e-3, max 10 epochs
- EarlyStopping (patience=5)
        """)
    with c2:
        st.markdown("""
#### Phase 2 — Fine-Tuning
- **Unfreeze** ทุก layer
- Train ด้วย LR ต่ำมาก (1e-5)
- max 20 epochs
- EarlyStopping + ModelCheckpoint
        """)

    # Data Aug
    st.header("4. Data Augmentation")
    st.markdown("""
| Augmentation | ค่าที่ใช้ |
|---|---|
| Rotation | ±20° |
| Width/Height Shift | 15% |
| Horizontal Flip | ✓ |
| Zoom | 15% |
| Brightness | 70%–130% |
    """)

    # History
    st.header("5. Training History")
    history_chart(hist)

    st.header("6. อ้างอิง")
    st.markdown("- [Sports Classification Dataset — Kaggle](https://www.kaggle.com/datasets/gpiosenka/sports-classification)")

# ════════════════════════════════════════════════════════════
# PAGE 3: ทดสอบ ML Model (Pokemon)
# ════════════════════════════════════════════════════════════

def page_test_ml(pokemon_df):
    st.title("🤖 ทดสอบ ML Ensemble Model")
    st.caption("🎮 อัปโหลดรูป Pokemon → โมเดลจะระบุชื่อ + Type + Evolution")
    st.markdown("---")

    model, le = load_ml_model()
    if model is None:
        st.error("❌ ยังไม่พบโมเดล ML — กรุณารัน: `python train_ml.py`")
        return
    st.success("✅ โหลดโมเดล ML สำเร็จ")

    st.markdown("### อัปโหลดรูปภาพ Pokemon")
    st.caption("รองรับ: JPG, JPEG, PNG")
    uploaded = st.file_uploader("เลือกรูป", type=["jpg","jpeg","png"], key="ml_up")

    if uploaded:
        col1, col2 = st.columns([1,3])
        with col1:
            st.image(uploaded, caption="รูปที่อัปโหลด", width=180)

        with st.spinner("🔍 กำลังวิเคราะห์..."):
            img  = preprocess_for_ml(uploaded)
            feat = extract_hog(img).reshape(1,-1)
            prob = model.predict_proba(feat)[0]
            idx  = int(np.argmax(prob))
            name = le.classes_[idx]

        info = lookup_pokemon(name, pokemon_df)

        st.markdown("---")
        st.subheader("🎯 ผลการวิเคราะห์")

        r1c1, r1c2 = st.columns([1,2])

        # Pokemon image from dataset
        with r1c1:
            pimg_path = os.path.join(POKE_IMGS, f"{name}.png")
            if os.path.exists(pimg_path):
                pimg = Image.open(pimg_path).convert("RGBA")
                bg = Image.new("RGBA", pimg.size, (255,255,255,255))
                bg.paste(pimg, mask=pimg.split()[3])
                st.image(bg.convert("RGB"), caption=f"Pokemon: {info['name']}",
                         width=150)

        with r1c2:
            st.markdown(f"## {info['name']}")
            st.markdown("**ประเภท (Type):**")
            tc = st.columns(4)
            with tc[0]:
                if info["type1"]: type_badge(info["type1"])
            with tc[1]:
                if info["type2"]: type_badge(info["type2"])
            st.markdown("")
            if info["evolution"]:
                evo = lookup_pokemon(info["evolution"], pokemon_df)
                st.markdown(f"**Evolution ถัดไป →** 🔆 **{evo['name']}**")
                ec = st.columns(4)
                with ec[0]:
                    if evo["type1"]: type_badge(evo["type1"])
                with ec[1]:
                    if evo["type2"]: type_badge(evo["type2"])
            else:
                st.markdown("**Evolution:** ไม่มี Evolution ต่อ 🏁")

        st.markdown("---")
        st.markdown("#### 📊 Top 5 ความน่าจะเป็น")
        bar_chart(prob, le.classes_, top_k=5, color="#8B63FF")

        # Confidence info
        conf = float(prob[idx])*100
        if conf >= 60:
            st.success(f"ระดับความมั่นใจ: **{conf:.1f}%** ✅")
        elif conf >= 30:
            st.warning(f"ระดับความมั่นใจ: **{conf:.1f}%** ⚠️ (ปานกลาง)")
        else:
            st.error(f"ระดับความมั่นใจ: **{conf:.1f}%** ❌ (ต่ำ)")

# ════════════════════════════════════════════════════════════
# PAGE 4: ทดสอบ Neural Network (Sports)
# ════════════════════════════════════════════════════════════

def page_test_nn(info):
    st.title("🧠 ทดสอบ Neural Network (EfficientNetB0)")
    st.caption("⚽ อัปโหลดรูปกีฬา → โมเดลจะจำแนกประเภทกีฬา")
    st.markdown("---")

    model = load_nn_model()
    if model is None:
        st.error("❌ ยังไม่พบโมเดล NN — กรุณารัน: `python train_nn.py`")
        return
    st.success("✅ โหลดโมเดล Neural Network สำเร็จ")

    classes = info.get("nn_classes", [])
    if not classes:
        # fallback: ดึงจาก folder structure
        if os.path.isdir(SPORTS_TRAIN):
            classes = sorted([d for d in os.listdir(SPORTS_TRAIN)
                              if os.path.isdir(os.path.join(SPORTS_TRAIN,d))])
    if not classes:
        st.error("❌ ไม่พบ class names")
        return

    st.markdown("### อัปโหลดรูปภาพกีฬา")
    st.caption("รองรับ: JPG, JPEG, PNG")
    uploaded = st.file_uploader("เลือกรูป", type=["jpg","jpeg","png"], key="nn_up")

    if uploaded:
        col1, col2 = st.columns([1,3])
        with col1:
            st.image(uploaded, caption="รูปที่อัปโหลด", width=180)

        with st.spinner("🔍 กำลังวิเคราะห์..."):
            img_in = preprocess_for_nn(uploaded)
            prob   = model.predict(img_in, verbose=0)[0]
            idx    = int(np.argmax(prob))
            label  = classes[idx] if idx < len(classes) else f"class_{idx}"

        st.markdown("---")
        st.subheader("🎯 ผลการวิเคราะห์")

        col_a, col_b = st.columns([1,2])
        with col_a:
            # แสดงรูปตัวอย่างจาก training set (ถ้ามี)
            sample_dir = os.path.join(SPORTS_TRAIN, label)
            sample_shown = False
            if os.path.isdir(sample_dir):
                samples = [f for f in os.listdir(sample_dir)
                           if f.lower().endswith((".jpg",".jpeg",".png"))]
                if samples:
                    sp = os.path.join(sample_dir, samples[0])
                    st.image(sp, caption=f"ตัวอย่าง: {label.title()}", width=150)
                    sample_shown = True
            if not sample_shown:
                st.markdown(f"### 🏅 {label.title()}")

        with col_b:
            st.markdown(f"## 🏅 {label.title()}")
            conf = float(prob[idx])*100
            if conf >= 70:
                st.success(f"ประเภทกีฬา: **{label.title()}**  \nความมั่นใจ: **{conf:.1f}%** ✅")
            elif conf >= 40:
                st.warning(f"ประเภทกีฬา: **{label.title()}**  \nความมั่นใจ: **{conf:.1f}%** ⚠️")
            else:
                st.error(f"ประเภทกีฬา: **{label.title()}**  \nความมั่นใจ: **{conf:.1f}%** ❌ (ต่ำ)")

        st.markdown("---")
        st.markdown("#### 📊 Top 5 ความน่าจะเป็น")
        bar_chart(prob, classes, top_k=5, color="#FF6584")

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Intelligent System Classification",
        page_icon="🤖",
        layout="wide",
    )

    pokemon_df = load_pokemon_df()
    info       = load_model_info()
    hist       = load_history()

    ml_acc = info.get("ml_accuracy","N/A")
    nn_acc = info.get("nn_accuracy","N/A")

    with st.sidebar:
        st.markdown("# 🤖 Intelligent System")
        st.markdown("---")

        st.markdown("**เลือกหน้า**")
        page = st.radio("", [
            "อธิบาย ML Model",
            "อธิบาย Neural Network",
            "ทดสอบ ML Model",
            "ทดสอบ Neural Network",
        ], label_visibility="collapsed")

        st.markdown("---")
        st.markdown(
            f"""
<div style="background:linear-gradient(135deg,#f0f4ff,#fdf0ff);
padding:16px;border-radius:12px;font-size:13px;line-height:1.8">
<b>📂 ML Dataset:</b><br>
&ensp;Pokemon ({info.get('ml_num_classes',809)} ตัว)<br>
<b>🎯 ML Accuracy:</b>
<span style="color:#6C63FF;font-weight:700">&ensp;{ml_acc}%</span><br><br>
<b>📂 NN Dataset:</b><br>
&ensp;Sports ({info.get('nn_num_classes',100)} ประเภท)<br>
<b>🎯 NN Accuracy:</b>
<span style="color:#FF6584;font-weight:700">&ensp;{nn_acc}%</span>
</div>""",
            unsafe_allow_html=True,
        )

    if   page == "อธิบาย ML Model":        page_explain_ml(info)
    elif page == "อธิบาย Neural Network":  page_explain_nn(info, hist)
    elif page == "ทดสอบ ML Model":         page_test_ml(pokemon_df)
    elif page == "ทดสอบ Neural Network":   page_test_nn(info)

if __name__ == "__main__":
    main()
