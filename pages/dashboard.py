import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Dashboard — AML Detection", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(145deg, #1A1D23, #22262E);
        border: 1px solid #2A2E35;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: border-color 0.3s;
    }
    .metric-card:hover { border-color: #E63946; }
    .metric-num { font-size: 2.2rem; font-weight: 800; color: #E63946; }
    .metric-lbl { font-size: 0.82rem; color: #888; margin-top: 0.3rem; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #E63946;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2A2E35;
    }
    .card {
        background: linear-gradient(145deg, #1A1D23, #22262E);
        border: 1px solid #2A2E35;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .card:hover { border-color: #E63946; }
    .bar-row { margin: 0.6rem 0; }
    .bar-header { display: flex; justify-content: space-between; margin-bottom: 4px; }
    .bar-name { font-size: 0.85rem; color: #DDD; }
    .bar-val { font-size: 0.85rem; color: #AAA; font-family: monospace; }
    .bar-bg { background: #2A2E35; border-radius: 6px; height: 10px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 6px; }
    .cm-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .cm-table th {
        background: #1A1D23;
        color: #E63946;
        padding: 8px 12px;
        text-align: center;
        border: 1px solid #2A2E35;
        font-weight: 600;
    }
    .cm-table td {
        padding: 8px 12px;
        text-align: center;
        border: 1px solid #2A2E35;
        color: #CCC;
    }
    .cm-highlight {
        background: rgba(230, 57, 70, 0.15);
        color: #E63946;
        font-weight: 700;
    }
    .insight-card {
        background: rgba(230, 57, 70, 0.06);
        border: 1px solid #2A2E35;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .insight-card p { color: #CCC; font-size: 0.88rem; line-height: 1.7; margin: 0; }
    .model-winner {
        background: rgba(46, 204, 113, 0.08);
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📊 Model Performance Dashboard")
st.markdown("Evaluation results of the tuned XGBoost multimodal classifier on the held-out test set.")
st.markdown("---")

cols = st.columns(4)
metrics = [
    ("84.21%", "Accuracy"),
    ("80.72%", "Macro F1"),
    ("83.89%", "Weighted F1"),
    ("38", "Test Patients"),
]

for col, (num, label) in zip(cols, metrics):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-num">{num}</div>'
        f'<div class="metric-lbl">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Per-Class Performance</div>', unsafe_allow_html=True)

class_data = [
    ("Control", 0.92, 1.00, 0.96, "#2ECC71"),
    ("CBFB_MYH11", 0.89, 1.00, 0.94, "#3498DB"),
    ("NPM1", 0.83, 0.71, 0.77, "#E67E22"),
    ("PML_RARA", 1.00, 0.60, 0.75, "#9B59B6"),
    ("RUNX1_RUNX1T1", 0.57, 0.67, 0.62, "#E74C3C"),
]

left, right = st.columns(2)

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**F1 Score by Class**")
    for name, prec, rec, f1, color in class_data:
        st.markdown(
            f'<div class="bar-row">'
            f'<div class="bar-header">'
            f'<span class="bar-name">{name}</span>'
            f'<span class="bar-val">{f1:.2f}</span></div>'
            f'<div class="bar-bg">'
            f'<div class="bar-fill" style="width: {f1*100}%; background: {color};"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Precision & Recall**")

    report_data = {
        "Class": [c[0] for c in class_data],
        "Precision": [c[1] for c in class_data],
        "Recall": [c[2] for c in class_data],
        "F1-Score": [c[3] for c in class_data],
        "Support": [12, 8, 7, 5, 6],
    }
    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)

labels = ["Control", "NPM1", "PML_RARA", "RUNX1_RUNX1T1", "CBFB_MYH11"]
cm = [
    [12, 0, 0, 0, 0],
    [0, 5, 0, 2, 0],
    [1, 0, 3, 1, 0],
    [0, 1, 0, 4, 1],
    [0, 0, 0, 0, 8],
]

header = "<tr><th>Actual \\ Predicted</th>"
for l in labels:
    header += f"<th>{l}</th>"
header += "</tr>"

rows = ""
for i, label in enumerate(labels):
    rows += f"<tr><th>{label}</th>"
    for j in range(5):
        val = cm[i][j]
        css = ' class="cm-highlight"' if i == j else ""
        rows += f"<td{css}>{val}</td>"
    rows += "</tr>"

st.markdown(
    f'<div class="card"><table class="cm-table">{header}{rows}</table></div>',
    unsafe_allow_html=True,
)

st.markdown("---")
st.markdown('<div class="section-header">Model Comparison (5-Fold CV)</div>', unsafe_allow_html=True)

models = [
    ("XGBoost", 76.84, 72.91, 76.52, True),
    ("Random Forest", 70.86, 63.09, 69.61, False),
    ("MLP", 70.24, 64.49, 70.21, False),
]

for name, acc, f1, wf1, winner in models:
    extra_class = " model-winner" if winner else ""
    badge = ' <span style="color: #2ECC71; font-size: 0.78rem;">✓ Selected</span>' if winner else ""
    st.markdown(
        f'<div class="insight-card{extra_class}">'
        f'<p><strong style="color: #FFF;">{name}</strong>{badge}<br>'
        f'Accuracy: {acc}% · Macro F1: {f1}% · Weighted F1: {wf1}%</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown('<div class="section-header">Key Observations</div>', unsafe_allow_html=True)

observations = [
    ("🟢", "Control", "Highest F1 (0.96) — the model reliably identifies healthy samples with perfect recall."),
    ("🔵", "CBFB_MYH11", "Strong performance with F1 of 0.94 and perfect recall (1.00)."),
    ("🟠", "NPM1", "Good precision (0.83) but recall drops to 0.71 — some NPM1 cases missed."),
    ("🟣", "PML_RARA", "Perfect precision but lower recall (0.60) — conservative in predicting this rare class."),
    ("🔴", "RUNX1_RUNX1T1", "Lowest F1 (0.62) — confused with NPM1 and CBFB_MYH11 due to smaller sample size."),
]

for icon, name, text in observations:
    st.markdown(
        f'<div class="insight-card">'
        f'<p>{icon} <strong style="color: #FFF;">{name}:</strong> {text}</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("Metrics computed on held-out test set · 5-fold stratified CV · ResNet50 + XGBoost pipeline")
