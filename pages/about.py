import streamlit as st

st.set_page_config(page_title="About — AML Detection", page_icon="ℹ️", layout="wide")

st.markdown("""
<style>
    .about-card {
        background: linear-gradient(145deg, #1A1D23, #22262E);
        border: 1px solid #2A2E35;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s;
    }
    .about-card:hover { border-color: #E63946; }
    .about-card h3 {
        color: #E63946;
        margin-top: 0;
        font-size: 1.1rem;
    }
    .about-card p, .about-card li {
        color: #CCCCCC;
        font-size: 0.9rem;
        line-height: 1.7;
    }
    .hero-section {
        text-align: center;
        padding: 2rem 0;
    }
    .hero-section h1 {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #E63946, #F1FAEE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-section p {
        color: #A8DADC;
        font-size: 1.05rem;
        max-width: 700px;
        margin: 0.5rem auto;
    }
    .stat-card {
        background: #1A1D23;
        border: 1px solid #2A2E35;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #E63946;
    }
    .stat-label {
        font-size: 0.82rem;
        color: #888;
    }
    .tech-badge {
        display: inline-block;
        background: rgba(230, 57, 70, 0.12);
        color: #E63946;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
    }
    .team-card {
        background: linear-gradient(145deg, #1A1D23, #22262E);
        border: 1px solid #2A2E35;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .pipeline-step {
        background: #1A1D23;
        border: 1px solid #2A2E35;
        border-radius: 8px;
        padding: 10px 16px;
        text-align: center;
        color: #CCCCCC;
        font-size: 0.85rem;
    }
    .arrow-text {
        color: #E63946;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #E63946;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2A2E35;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-section">
    <h1>🔬 AML Leukemia Detection System</h1>
    <p>A multimodal deep learning application combining blood cell microscopy 
    images with clinical data to classify Acute Myeloid Leukemia subtypes</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown('<div class="section-title">📊 Dataset</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, num, label in zip(
    [c1, c2, c3, c4],
    ["189", "81,214", "~430", "5"],
    ["Patients", "Cell Images", "Images/Patient", "Classes"]
):
    col.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{num}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="about-card">
    <h3>TCIA AML-Cytomorphology Dataset</h3>
    <p>Hehr et al., PLOS Digital Health (2023)<br><br>
    Single-cell microscopy images of peripheral blood smears from patients with 
    various AML genetic subtypes and healthy controls, alongside clinical 
    differential blood count data from the same patients.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="section-title">⚙️ Technical Pipeline</div>', unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    st.markdown("""
    <div class="about-card">
        <h3>🖼️ Image Branch</h3>
        <p>
        1. Blood smear images resized to 224×224<br>
        2. Feature extraction via pretrained <b>ResNet50</b><br>
        3. 2,048-dim feature vector per image<br>
        4. Mean-pooling across patient images<br>
        5. StandardScaler → PCA (30 components)
        </p>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="about-card">
        <h3>📋 Clinical Branch</h3>
        <p>
        1. 15 features from peripheral blood differential<br>
        2. Age, sex, leucocyte count<br>
        3. Myeloblast, promyelocyte, myelocyte %<br>
        4. Neutrophil, eosinophil, basophil %<br>
        5. StandardScaler normalization
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="about-card">
    <h3>🔗 Fusion & Classification</h3>
    <p>PCA image features (30-dim) + scaled clinical features (15-dim) are concatenated 
    into a 45-dimensional vector, then classified by a <b>tuned XGBoost</b> multiclass 
    classifier into 5 AML subtypes with probability scores.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

p1, a1, p2, a2, p3, a3, p4, a4, p5 = st.columns([2, 1, 2, 1, 2, 1, 2, 1, 2])
steps = [
    (p1, "Images (.tif)"),
    (p2, "ResNet50"),
    (p3, "Scaler + PCA"),
    (p4, "Concatenate"),
    (p5, "XGBoost"),
]
arrows = [a1, a2, a3, a4]

for (col, text) in steps:
    col.markdown(f'<div class="pipeline-step">{text}</div>', unsafe_allow_html=True)
for arrow in arrows:
    arrow.markdown('<div class="arrow-text">→</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="section-title">🏷️ Classification Classes</div>', unsafe_allow_html=True)

classes = [
    ("Control", "Healthy / No AML detected", "🟢", "N/A"),
    ("NPM1", "Nucleophosmin 1 mutation", "🔵", "Favorable"),
    ("PML_RARA", "Acute Promyelocytic Leukemia (APL)", "🟣", "Highly treatable"),
    ("RUNX1_RUNX1T1", "t(8;21) translocation AML", "🔴", "Favorable"),
    ("CBFB_MYH11", "Core Binding Factor AML, inv(16)", "🟠", "Favorable"),
]

for name, desc, icon, prognosis in classes:
    st.markdown(f"""
    <div class="about-card" style="display: flex; align-items: center; gap: 1rem;">
        <div style="font-size: 1.5rem;">{icon}</div>
        <div style="flex: 1;">
            <h3 style="margin-bottom: 0.2rem;">{name}</h3>
            <p style="margin: 0;">{desc}</p>
        </div>
        <div>
            <span class="tech-badge">{prognosis}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="section-title">🛠️ Technologies</div>', unsafe_allow_html=True)

techs = [
    "TensorFlow", "ResNet50", "XGBoost", "Scikit-learn",
    "Streamlit", "Python", "Groq API", "Llama 3.1",
    "Pandas", "NumPy", "TCIA Dataset", "Pillow",
]

badges_html = "".join([f'<span class="tech-badge">{t}</span>' for t in techs])
st.markdown(f'<div style="text-align: center;">{badges_html}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="section-title">👥 Team</div>', unsafe_allow_html=True)

st.markdown("""
<div class="team-card">
    <h3 style="color: #E63946; font-size: 1.2rem;">NIBM KIC</h3>
    <p style="color: #A8DADC; font-size: 1rem; margin: 0.5rem 0;">Computer Science with AI</p>
    <p style="color: #888; font-size: 0.9rem;">Team Code: KIC-HNDCSAI-252F</p>
    <p style="color: #CCCCCC; font-size: 0.95rem;">Members: 001 · 004 · 026 · 030</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.warning(
    "⚠️ This tool is for research and educational purposes only. "
    "It is NOT a substitute for professional medical diagnosis."
)
st.caption("Final Year Project · NIBM KIC · Computer Science with AI · 2025")