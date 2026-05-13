import streamlit as st

st.set_page_config(
    page_title="AML Leukemia Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #E63946, #F1FAEE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #A8DADC;
        margin-bottom: 2rem;
        line-height: 1.7;
    }
    .card {
        background: linear-gradient(145deg, #1A1D23, #22262E);
        border: 1px solid #2A2E35;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s;
    }
    .card:hover { border-color: #E63946; }
    .card h3 { color: #E63946; margin-top: 0; font-size: 1.05rem; }
    .card p { color: #CCC; font-size: 0.9rem; line-height: 1.7; }
    .stat-box {
        background: #1A1D23;
        border: 1px solid #2A2E35;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-num { font-size: 2rem; font-weight: 700; color: #E63946; }
    .stat-lbl { font-size: 0.8rem; color: #888; margin-top: 0.2rem; }
    .step-box {
        background: #1A1D23;
        border: 1px solid #2A2E35;
        border-radius: 8px;
        padding: 8px 14px;
        text-align: center;
        color: #CCC;
        font-size: 0.82rem;
    }
    .step-arrow { color: #E63946; font-weight: 700; font-size: 1.1rem; text-align: center; }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #151921 100%);
        border-right: 1px solid #2A2E35;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {
        color: #A8DADC;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #E63946;
    }
    [data-testid="stSidebar"] .stCaption p {
        color: #555;
    }
            
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔬 AML Detection System")
    st.markdown("---")
    st.markdown("""
    **Pages**
    - 🏠 Home — Overview
    - 🔬 Prediction — Upload & Predict
    - 🤖 AI Chatbot — Ask about AML
    - 📊 Dashboard — Model Metrics
    - ℹ️ About — Project Info
    """)
    st.markdown("---")
    st.caption("ResNet50 + XGBoost")
    st.caption("TCIA AML-Cytomorphology")

st.markdown('<p class="hero-title">AML Leukemia Detection System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">'
    'A multimodal AI system combining blood cell microscopy images with '
    'clinical data to classify Acute Myeloid Leukemia subtypes.'
    '</p>',
    unsafe_allow_html=True,
)

cols = st.columns(4)
stats = [("5", "AML Classes"), ("15", "Clinical Features"), ("2048", "Image Features"), ("XGBoost", "Classifier")]

for col, (num, label) in zip(cols, stats):
    col.markdown(
        f'<div class="stat-box">'
        f'<div class="stat-num">{num}</div>'
        f'<div class="stat-lbl">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
steps = ["Images (.tif)", "ResNet50", "Scaler + PCA", "Concat", "XGBoost", "Prediction"]

pipeline_cols = st.columns(len(steps) * 2 - 1)
for i, step in enumerate(steps):
    pipeline_cols[i * 2].markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)
    if i < len(steps) - 1:
        pipeline_cols[i * 2 + 1].markdown('<div class="step-arrow">→</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
left, right = st.columns(2)

with left:
    st.markdown("""
    <div class="card">
        <h3>🧬 How It Works</h3>
        <p>
        <b>Image Branch</b> — Blood smear images are processed through a pretrained 
        ResNet50 network to extract 2,048 deep features per image. Multiple images 
        per patient are averaged, scaled, and reduced via PCA.<br><br>
        <b>Clinical Branch</b> — 15 clinical lab values including white blood cell 
        differential counts, age, sex, and leucocyte count are standardized.<br><br>
        <b>Fusion</b> — Both branches are concatenated and classified by XGBoost 
        for 5-class AML subtype prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>📊 Dataset</h3>
        <p>
        Built using the <b>TCIA AML-Cytomorphology</b> dataset — 81,214 single-cell 
        microscopy images from 189 patients with various AML genetic subtypes and 
        healthy controls, along with clinical differential blood count data.
        </p>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="card">
        <h3>🏷️ Classification Classes</h3>
        <p>
        • <b>Control</b> — Healthy / No AML detected<br>
        • <b>NPM1</b> — Nucleophosmin 1 mutation<br>
        • <b>PML_RARA</b> — Acute Promyelocytic Leukemia<br>
        • <b>RUNX1_RUNX1T1</b> — t(8;21) translocation AML<br>
        • <b>CBFB_MYH11</b> — Core Binding Factor AML
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🎯 Performance</h3>
        <p>
        The tuned XGBoost multimodal classifier achieves <b>84.21% accuracy</b> 
        on the held-out test set with a macro F1 score of <b>80.72%</b>. 
        The model was selected after comparing Random Forest, XGBoost, and MLP 
        classifiers through 5-fold stratified cross-validation.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.warning(
    "⚠️ This tool is for research and educational purposes only. "
    "It is NOT a substitute for professional medical diagnosis."
)
st.caption("Final Year Project · NIBM KIC · Computer Science with AI")

