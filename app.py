"""
app.py - Intelligent System Classification App
- ML Model  : Sports Ball Classification  (HOG + Random Forest)
- Neural Net: Sports Image Classification (MobileNetV2, 100 categories)
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
BALL_DIR   = os.path.join(BASE_DIR, "ball",   "train")
SPORTS_DIR = os.path.join(BASE_DIR, "sports", "train")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

ML_IMG_SIZE = 64
NN_IMG_SIZE = 128

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

def inject_css():
    st.markdown(
        "<style>"
        "html,body,[class*='css'],.stMarkdown,p,li,label"
        "{font-family:'Sarabun','Inter',sans-serif!important}"
        "h1,h2,h3,h4,h5{font-family:'Sarabun','Inter',sans-serif!important;font-weight:700!important}"
        "[data-testid='stSidebar']{background:linear-gradient(180deg,#1a1a2e,#16213e)!important}"
        "[data-testid='stSidebar'] *{color:#e0e0e0!important;font-family:'Sarabun',sans-serif!important}"
        "[data-testid='stSidebar'] h2{color:#fff!important}"
        ".nav-btn{display:block;width:100%;text-align:left;"
        "border:none;border-radius:8px;cursor:pointer;margin:3px 0;"
        "padding:9px 14px;color:#c8c8d4;background:transparent;font-size:14px;"
        "font-family:'Sarabun',sans-serif;transition:all 0.2s}"
        ".nav-btn:hover{background:rgba(255,255,255,0.1)!important;color:#fff!important}"
        "[data-testid='stSidebar'] .stButton button{text-align:left!important;"
        "background:transparent!important;border:none!important;"
        "color:#c8c8d4!important;font-size:14px!important;padding:9px 14px!important;"
        "border-radius:8px!important;margin:2px 0!important;font-weight:400!important}"
        "[data-testid='stSidebar'] .stButton button:hover{background:rgba(255,255,255,0.1)!important;"
        "color:#fff!important}"
        ".active-nav button{background:linear-gradient(135deg,#6C63FF,#9B5DE5)!important;"
        "color:#fff!important;font-weight:700!important}"
        "[data-testid='metric-container']{background:linear-gradient(135deg,#f8f9ff,#fff5f8);"
        "border-radius:12px;border:1px solid #e8e0ff;padding:16px!important}"
        "[data-testid='stMetricValue']{font-size:24px!important;font-weight:700!important;color:#6C63FF!important}"
        "table{border-collapse:collapse;width:100%;margin:12px 0}"
        "th{background:#6C63FF;color:white;padding:10px 14px;font-size:14px}"
        "td{padding:9px 14px;border-bottom:1px solid #eee;font-size:14px}"
        "tr:hover td{background:#f8f4ff}"
        ".ref-box{background:linear-gradient(135deg,#f0f4ff,#fff0f8);"
        "border-left:4px solid #6C63FF;border-radius:8px;"
        "padding:16px 20px;margin:8px 0;font-size:14px;line-height:1.9}"
        ".ref-box a{color:#6C63FF;text-decoration:none;font-weight:600}"
        ".ref-number{display:inline-block;background:#6C63FF;color:white;"
        "border-radius:50%;width:22px;height:22px;text-align:center;"
        "line-height:22px;font-size:12px;margin-right:8px;font-weight:700}"
        ".info-card{background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);"
        "border-radius:12px;padding:16px;font-size:13px;line-height:2}"
        ".info-card b{color:#fff}"
        ".acc-ml{color:#a78bfa;font-weight:700;font-size:15px}"
        ".acc-nn{color:#fb7185;font-weight:700;font-size:15px}"
        "[data-testid='stSidebarCollapseButton']{display:none!important}"
        "button[kind='header']{display:none!important}"
        "section[data-testid='stSidebar'] > div:first-child > div:first-child button{display:none!important}"
        "</style>",
        unsafe_allow_html=True,
    )

@st.cache_data
def load_model_info():
    p = os.path.join(MODEL_DIR, "model_info.json")
    return json.load(open(p)) if os.path.exists(p) else {}

@st.cache_data
def load_history():
    p = os.path.join(MODEL_DIR, "history.json")
    return json.load(open(p)) if os.path.exists(p) else {}

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
    try:
        import tensorflow as tf
        for name in ["model_nn.keras", "model_nn_best.keras"]:
            p = os.path.join(MODEL_DIR, name)
            if os.path.exists(p):
                return tf.keras.models.load_model(p)
    except ImportError:
        return None
    return None


def preprocess_ml(uploaded, size=ML_IMG_SIZE):
    img = Image.open(uploaded).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.array(img)

def preprocess_nn(uploaded, size=NN_IMG_SIZE):
    img = Image.open(uploaded).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def extract_hog(img):
    from skimage.feature import hog
    return hog(img, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), channel_axis=-1)

def sample_image(folder, label):
    path = os.path.join(folder, label)
    if not os.path.isdir(path):
        return None
    files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return os.path.join(path, files[0]) if files else None

def bar_chart(probs, labels, top_k=5, color="#6C63FF"):
    idx  = np.argsort(probs)[::-1][:top_k]
    vals = [probs[i] * 100 for i in idx]
    labs = [BALL_DISPLAY.get(str(labels[i]), str(labels[i]).replace("_", " ").title()) for i in idx]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.set_facecolor("#fafafa"); fig.patch.set_facecolor("#fafafa")
    cs = [color if i == 0 else "#DDD8F8" for i in range(len(labs))]
    ax.barh(labs[::-1], vals[::-1], color=cs[::-1], height=0.5, edgecolor="none")
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=10, color="#333")
    ax.set_xlim(0, 110)
    ax.set_xlabel("ความนาจะเปน (%)", fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

def training_history_chart(hist):
    if not hist:
        st.info("ไมพบขอมล training history")
        return
    phases = []
    if "p1_acc" in hist:
        phases.append(("Phase 1 - Train Top Layers", hist["p1_acc"], hist["p1_val_acc"], hist["p1_loss"], hist["p1_val_loss"]))
    if "p2_acc" in hist:
        phases.append(("Phase 2 - Fine-Tuning", hist["p2_acc"], hist["p2_val_acc"], hist["p2_loss"], hist["p2_val_loss"]))
    for title, ta, va, tl, vl in phases:
        st.markdown(f"**{title}**")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        fig.patch.set_facecolor("#fafafa")
        ep = range(1, len(ta) + 1)
        axes[0].plot(ep, ta, label="Train", color="#6C63FF", lw=2)
        axes[0].plot(ep, va, label="Val",   color="#FF6584", lw=2, ls="--")
        axes[0].set_title("Accuracy"); axes[0].legend()
        axes[0].spines[["top", "right"]].set_visible(False)
        axes[1].plot(ep, tl, label="Train", color="#6C63FF", lw=2)
        axes[1].plot(ep, vl, label="Val",   color="#FF6584", lw=2, ls="--")
        axes[1].set_title("Loss"); axes[1].legend()
        axes[1].spines[["top", "right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

def confidence_badge(conf):
    if conf >= 70:
        st.success(f"ความมั่นใจ: **{conf:.1f}%** (สูง)")
    elif conf >= 40:
        st.warning(f"ความมั่นใจ: **{conf:.1f}%** (ปานกลาง)")
    else:
        st.error(f"ความมั่นใจ: **{conf:.1f}%** (ต่ำ - ลองรูปอื่น)")

def references_ml():
    st.markdown("---")
    st.header("แหล่งอ้างอิง")
    st.markdown(
        '<div class="ref-box">'
        '<span class="ref-number">1</span>'
        '<b>Sports Balls Image Dataset</b> — Kaggle<br>'
        '&emsp;<a href="https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification" target="_blank">'
        'kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification</a>'
        '<br><span class="ref-number">2</span>'
        '<b>Dalal, N. &amp; Triggs, B. (2005)</b> — Histograms of Oriented Gradients for Human Detection. <i>IEEE CVPR 2005.</i><br>'
        '&emsp;<a href="https://ieeexplore.ieee.org/document/1467360" target="_blank">ieeexplore.ieee.org/document/1467360</a>'
        '<br><span class="ref-number">3</span>'
        '<b>Breiman, L. (2001)</b> — Random Forests. <i>Machine Learning, 45(1), 5-32.</i><br>'
        '&emsp;<a href="https://doi.org/10.1023/A:1010933404324" target="_blank">doi.org/10.1023/A:1010933404324</a>'
        '<br><span class="ref-number">4</span>'
        '<b>scikit-image: HOG Feature Descriptor</b><br>'
        '&emsp;<a href="https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html" target="_blank">'
        'scikit-image.org/docs/stable/...plot_hog.html</a>'
        '<br><span class="ref-number">5</span>'
        '<b>scikit-learn: RandomForestClassifier</b><br>'
        '&emsp;<a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html" target="_blank">'
        'scikit-learn.org/stable/modules/...RandomForestClassifier.html</a>'
        '</div>',
        unsafe_allow_html=True,
    )

def references_nn():
    st.markdown("---")
    st.header("แหล่งอ้างอิง")
    st.markdown(
        '<div class="ref-box">'
        '<span class="ref-number">1</span>'
        '<b>Sports Classification Dataset (100 ประเภท)</b> — Kaggle<br>'
        '&emsp;<a href="https://www.kaggle.com/datasets/gpiosenka/sports-classification" target="_blank">'
        'kaggle.com/datasets/gpiosenka/sports-classification</a>'
        '<br><span class="ref-number">2</span>'
        '<b>Sandler, M. et al. (2018)</b> — MobileNetV2: Inverted Residuals and Linear Bottlenecks. <i>IEEE CVPR 2018.</i><br>'
        '&emsp;<a href="https://arxiv.org/abs/1801.04381" target="_blank">arxiv.org/abs/1801.04381</a>'
        '<br><span class="ref-number">3</span>'
        '<b>Transfer Learning with TensorFlow/Keras</b><br>'
        '&emsp;<a href="https://www.tensorflow.org/tutorials/images/transfer_learning" target="_blank">'
        'tensorflow.org/tutorials/images/transfer_learning</a>'
        '<br><span class="ref-number">4</span>'
        '<b>Keras Documentation — MobileNetV2</b><br>'
        '&emsp;<a href="https://keras.io/api/applications/mobilenet/" target="_blank">'
        'keras.io/api/applications/mobilenet/</a>'
        '<br><span class="ref-number">5</span>'
        '<b>Streamlit Documentation</b><br>'
        '&emsp;<a href="https://docs.streamlit.io" target="_blank">docs.streamlit.io</a>'
        '</div>',
        unsafe_allow_html=True,
    )

def page_explain_ml(info):
    st.title("ML Model — Sports Ball Classification")
    st.caption("อัลกอริทึม: HOG Feature Extraction + Random Forest Classifier")
    st.markdown("---")

    st.header("1. ชุดข้อมูล (Dataset)")
    n_cls = info.get("ml_num_classes", 12)
    c1, c2, c3 = st.columns(3)
    c1.metric("จำนวนประเภทบอล", f"{n_cls} ประเภท")
    c2.metric("ขนาดรูปภาพ", f"{info.get('ml_img_size', 64)}x{info.get('ml_img_size', 64)} px")
    c3.metric("ความแม่นยำ (Test Set)", f"{info.get('ml_accuracy', '?')}%")

    classes = info.get("ml_classes", list(BALL_DISPLAY.keys()))
    st.markdown("**ประเภทบอลทั้งหมดในชุดข้อมูล:**")
    cols = st.columns(4)
    for i, cls in enumerate(classes):
        emoji = BALL_EMOJI.get(cls, "o")
        name  = BALL_DISPLAY.get(cls, cls.replace("_", " ").title())
        cols[i % 4].markdown(f"{emoji} {name}")

    st.header("2. การเตรียมข้อมูล (Data Augmentation)")
    st.markdown("""
| เทคนิค | รายละเอียด |
|--------|------------|
| Original | รูปต้นฉบับ |
| Flip Horizontal | กลับภาพซ้าย-ขวา |
| Rotation +15 degree | หมุนตามเข็มนาฬิกา 15 องศา |
| Rotation -15 degree | หมุนทวนเข็มนาฬิกา 15 องศา |
| รวม | 4 รูปต่อ 1 ภาพต้นฉบับ |
""")

    st.header("3. การสกัดลักษณะ (HOG Features)")
    st.markdown("""
**Histogram of Oriented Gradients (HOG)** คำนวณการกระจายทิศทางของ gradient ในแต่ละส่วนของภาพ

| พารามิเตอร์ | ค่าที่ใช้ |
|------------|----------|
| orientations | 9 ทิศทาง |
| pixels_per_cell | 8 x 8 พิกเซล |
| cells_per_block | 2 x 2 เซลล์ |
| ขนาด Feature Vector | ~2,916 ค่าต่อรูป |
""")

    st.header("4. โมเดล Random Forest")
    st.markdown("""
| พารามิเตอร์ | ค่า |
|------------|-----|
| n_estimators | 150 ต้น |
| Voting | Soft (ใช้ probability) |
| n_jobs | -1 (ทุก CPU core) |
""")

    st.header("5. ผลลัพธ์")
    st.metric("ความแม่นยำบน Test Set", f"{info.get('ml_accuracy', '?')}%")
    references_ml()

def page_explain_nn(info, hist):
    st.title("Neural Network — Sports Classification")
    st.caption("อัลกอริทึม: Transfer Learning ด้วย MobileNetV2 (Pre-trained ImageNet)")
    st.markdown("---")

    st.header("1. ชุดข้อมูล (Dataset)")
    n_cls = info.get("nn_num_classes", 100)
    c1, c2, c3 = st.columns(3)
    c1.metric("จำนวนประเภทกีฬา", f"{n_cls} ประเภท")
    c2.metric("ขนาดรูปภาพ", f"{info.get('nn_img_size', 128)}x{info.get('nn_img_size', 128)} px")
    c3.metric("ความแม่นยำ (Val Set)", f"{info.get('nn_accuracy', '?')}%")

    classes = info.get("nn_classes", [])
    if classes:
        st.markdown("**ตัวอย่างประเภทกีฬา (20 แรก):**")
        cols = st.columns(5)
        for i, cls in enumerate(classes[:20]):
            cols[i % 5].markdown(f"- {cls.replace('_',' ').title()}")

    st.header("2. สถาปัตยกรรมโมเดล (MobileNetV2)")
    st.markdown("""
ใช้เทคนิค **Transfer Learning** จาก MobileNetV2 (pre-trained ImageNet):

| Layer | รายละเอียด |
|-------|-----------|
| Input | 128 x 128 x 3 |
| MobileNetV2 Base | Pre-trained (ImageNet weights) |
| GlobalAveragePooling2D | ลด dimension |
| BatchNormalization | Normalize |
| Dense(256, ReLU) | Fully Connected |
| Dropout(0.4) | ป้องกัน Overfitting |
| Dense(N, Softmax) | Output Layer |
""")

    st.header("3. กลยุทธ์การฝึกสอน 2 Phase")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Phase 1 — Feature Extraction**
| รายการ | ค่า |
|--------|-----|
| MobileNetV2 | Frozen |
| Learning Rate | 1e-3 |
| Epochs | 5 |
""")
    with c2:
        st.markdown("""
**Phase 2 — Fine-Tuning**
| รายการ | ค่า |
|--------|-----|
| 30 layers สุดท้าย | Unfrozen |
| Learning Rate | 1e-5 |
| Epochs | 10 |
""")

    st.header("4. Training History")
    training_history_chart(hist)

    st.header("5. ผลลัพธ์")
    st.metric("ความแม่นยำบน Validation Set", f"{info.get('nn_accuracy', '?')}%")
    references_nn()

def page_test_ml():
    st.title("ทดสอบ ML Model — Sports Ball")
    st.caption("อัปโหลดรูปบอลกีฬา แล้วโมเดลจะวิเคราะห์ว่าเป็นบอลประเภทใด")
    st.markdown("---")

    model, le = load_ml_model()
    if model is None:
        st.error("ยังไมพบโมเดล ML กรณารน: python train_ml.py")
        return
    st.success("โหลดโมเดล Random Forest สำเร็จ")

    with st.expander("ดูตัวอย่างรูปบอลแต่ละประเภทจาก Dataset"):
        gcols = st.columns(6)
        for i, cls in enumerate(BALL_DISPLAY.keys()):
            sp = sample_image(BALL_DIR, cls)
            if sp:
                gcols[i % 6].image(sp,
                    caption=f"{BALL_EMOJI.get(cls,'o')} {BALL_DISPLAY.get(cls,cls)}",
                    width=90)

    st.markdown("### อัปโหลดรูปบอลกีฬา")
    uploaded = st.file_uploader("รองรับไฟล์ JPG, JPEG, PNG",
                                type=["jpg", "jpeg", "png"], key="ml_up")
    if uploaded:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded, caption="รูปที่อัปโหลด", use_column_width=True)

        with st.spinner("กำลังสกัด HOG features และวิเคราะห์..."):
            img  = preprocess_ml(uploaded)
            feat = extract_hog(img).reshape(1, -1)
            prob = model.predict_proba(feat)[0]
            idx  = int(np.argmax(prob))
            label = le.classes_[idx]

        emoji   = BALL_EMOJI.get(label, "o")
        display = BALL_DISPLAY.get(label, label.replace("_", " ").title())
        conf    = float(prob[idx]) * 100

        with col2:
            st.markdown(f"## {emoji} {display}")
            confidence_badge(conf)
            sp = sample_image(BALL_DIR, label)
            if sp:
                st.image(sp, caption=f"ตัวอย่างจาก Dataset: {display}", width=140)

        st.markdown("---")
        st.markdown("#### Top 5 ความน่าจะเป็น (Probability)")
        bar_chart(prob, le.classes_, top_k=5, color="#6C63FF")

def page_test_nn(info):
    st.title("ทดสอบ Neural Network — Sports")
    st.caption("อัปโหลดรูปกีฬา แล้วโมเดลจะวิเคราะห์ว่าเป็นกีฬาประเภทใด")
    st.markdown("---")

    model = load_nn_model()
    if model is None:
        st.error("ยังไมพบโมเดล NN กรณารน: python train_nn.py")
        return
    st.success("โหลดโมเดล MobileNetV2 สำเร็จ")

    classes = info.get("nn_classes", [])
    if not classes and os.path.isdir(SPORTS_DIR):
        classes = sorted([d for d in os.listdir(SPORTS_DIR)
                          if os.path.isdir(os.path.join(SPORTS_DIR, d))])

    st.markdown("### อัปโหลดรูปกีฬา")
    uploaded = st.file_uploader("รองรับไฟล์ JPG, JPEG, PNG",
                                type=["jpg", "jpeg", "png"], key="nn_up")
    if uploaded:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded, caption="รูปที่อัปโหลด", use_column_width=True)

        with st.spinner("กำลังประมวลผลด้วย Neural Network..."):
            img_in = preprocess_nn(uploaded)
            prob   = model.predict(img_in, verbose=0)[0]
            idx    = int(np.argmax(prob))
            label  = classes[idx] if idx < len(classes) else f"class_{idx}"

        display = label.replace("_", " ").title()
        conf    = float(prob[idx]) * 100

        with col2:
            st.markdown(f"## {display}")
            confidence_badge(conf)
            sp = sample_image(SPORTS_DIR, label)
            if sp:
                st.image(sp, caption=f"ตัวอย่างจาก Dataset: {display}", width=140)

        st.markdown("---")
        st.markdown("#### Top 5 ความน่าจะเป็น (Probability)")
        bar_chart(prob, classes, top_k=5, color="#FF6584")

def main():
    st.set_page_config(
        page_title="Intelligent System — Sports Classification",
        page_icon="🎱",
        layout="wide",
    )
    inject_css()

    info = load_model_info()
    hist = load_history()

    PAGES = [
        ("อธิบาย ML Model",       "📊"),
        ("อธิบาย Neural Network",  "🧠"),
        ("ทดสอบ ML (บอล)",        "🎱"),
        ("ทดสอบ NN (กีฬา)",       "🏅"),
    ]
    if "page" not in st.session_state:
        st.session_state.page = "อธิบาย ML Model"

    with st.sidebar:
        st.markdown("## Intelligent System")
        st.markdown("##### Sports Image Classification")
        st.markdown("---")

        for label, icon in PAGES:
            is_active = (st.session_state.page == label)
            arrow = "▶" if is_active else "›"
            prefix = f"{arrow}  {icon} "
            wrap_cls = "active-nav" if is_active else ""
            st.markdown(f'<div class="{wrap_cls}">', unsafe_allow_html=True)
            if st.button(f"{prefix}{label}", key=f"nav_{label}", use_container_width=True):
                st.session_state.page = label
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        page = st.session_state.page

        st.markdown("---")
        st.markdown(
            '<div class="info-card">'
            f'<b>ML Dataset</b><br>'
            f'Sports Balls — {info.get("ml_num_classes", 12)} ประเภท<br>'
            f'<b>ML Accuracy</b><br>'
            f'<span class="acc-ml">{info.get("ml_accuracy", "?")}%</span>'
            '<hr style="border:0;border-top:1px solid rgba(255,255,255,0.1);margin:10px 0">'
            f'<b>NN Dataset</b><br>'
            f'Sports — {info.get("nn_num_classes", 100)} ประเภท<br>'
            f'<b>NN Accuracy</b><br>'
            f'<span class="acc-nn">{info.get("nn_accuracy", "?")}%</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:11px;color:#888;text-align:center">'
            'Intelligent System Project<br>KMUTNB 2026</div>',
            unsafe_allow_html=True,
        )

    if   "ML Model"       in page: page_explain_ml(info)
    elif "Neural Network" in page: page_explain_nn(info, hist)
    elif "ML" in page:             page_test_ml()
    elif "NN" in page:             page_test_nn(info)

if __name__ == "__main__":
    main()
