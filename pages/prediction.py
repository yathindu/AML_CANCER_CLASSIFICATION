import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import tempfile

# Configure Streamlit page settings
st.set_page_config(page_title="Prediction — AML Detection", page_icon="🔬", layout="wide")

# Custom CSS styling for the app
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #E63946;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2A2E35;
    }
    .result-card {
        background: linear-gradient(145deg, #1A1D23, #22262E);
        border: 2px solid #2A2E35;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .result-conf {
        font-size: 1.1rem;
        color: #A8DADC;
    }
    .info-card {
        background: linear-gradient(145deg, #1A1D23, #22262E);
        border: 1px solid #2A2E35;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .info-card p { color: #CCC; font-size: 0.88rem; line-height: 1.7; }
    .prob-row {
        margin: 0.5rem 0;
    }
    .prob-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 3px;
    }
    .prob-name { font-size: 0.85rem; color: #DDD; }
    .prob-val { font-size: 0.85rem; color: #AAA; font-family: monospace; }
    .prob-bg {
        background: #2A2E35;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.8s ease-out;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔬 AML Prediction")
st.markdown("Upload peripheral blood smear images and enter clinical data to classify the AML subtype.")
st.markdown("---")

# Path to model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

CLINICAL_FEATURES = [
    "sex_1f_2m", "age", "leucocytes_per_ul",
    "pb_myeloblast", "pb_promyelocyte", "pb_myelocyte", "pb_metamyelocyte",
    "pb_neutrophil_band", "pb_neutrophil_segmented",
    "pb_eosinophil", "pb_basophil", "pb_monocyte",
    "pb_lymph_typ", "pb_lymph_atyp_react", "pb_other",
]


LABEL_MAP = {
    0: "Control",
    1: "NPM1",
    2: "PML_RARA",
    3: "RUNX1_RUNX1T1",
    4: "CBFB_MYH11",
}

LABEL_COLORS = {
    "Control": "#2ECC71",
    "NPM1": "#E67E22",
    "PML_RARA": "#9B59B6",
    "RUNX1_RUNX1T1": "#E74C3C",
    "CBFB_MYH11": "#3498DB",
}

LABEL_DESCRIPTIONS = {
    "Control": "No AML detected — sample appears consistent with healthy blood.",
    "NPM1": "Nucleophosmin 1 mutation — one of the most common AML mutations. Often favorable prognosis.",
    "PML_RARA": "Acute Promyelocytic Leukemia (APL) — highly treatable with ATRA-based therapy.",
    "RUNX1_RUNX1T1": "AML with t(8;21) translocation — generally favorable prognosis with chemotherapy.",
    "CBFB_MYH11": "Core Binding Factor AML — associated with inv(16). Generally favorable prognosis.",
}

# Load preprocessing pipeline and trained model
@st.cache_resource
def load_pipeline():
    try:
        preprocessing = joblib.load(os.path.join(MODEL_DIR, "preprocessing.pkl"))
        xgb_model = joblib.load(os.path.join(MODEL_DIR, "final_XGboost_model.pkl"))
        scaler_img = preprocessing["scaler_img"]
        pca = preprocessing["pca"]
        scaler_clinical = preprocessing["scaler_clinical"]
        return xgb_model, scaler_img, pca, scaler_clinical
    except FileNotFoundError:
        return None

# Load pretrained ResNet50 model for feature extraction
@st.cache_resource
def load_resnet():
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    base_model.trainable = False
    return base_model

# Extract image features using ResNet50
def extract_image_features(image_files, base_model):
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.resnet50 import preprocess_input

    all_features = []
    for img_file in image_files:
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_file.getvalue())
            tmp_path = tmp.name

        # Load and preprocess image
        img = keras_image.load_img(tmp_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract deep features
        feat = base_model.predict(img_array, verbose=0)
        all_features.append(feat.flatten())
        os.unlink(tmp_path)

    all_features = np.array(all_features)
    patient_feature = np.mean(all_features, axis=0).reshape(1, -1)
    return patient_feature

# Predict AML subtype using image + clinical data
def predict_patient(image_features, clinical_dict, xgb_model, scaler_img, pca, scaler_clinical):
    img_scaled = scaler_img.transform(image_features) # Scale image features
    img_pca = pca.transform(img_scaled) # Apply PCA

    # Prepare clinical data
    clinical_df = pd.DataFrame([clinical_dict])
    clinical_df = clinical_df[CLINICAL_FEATURES]
    clinical_scaled = scaler_clinical.transform(clinical_df.values)

    # Combine image and clinical features
    X_final = np.concatenate([img_pca, clinical_scaled], axis=1)

    # Make prediction
    pred_class = int(xgb_model.predict(X_final)[0])
    pred_prob = xgb_model.predict_proba(X_final)[0]
    pred_label = LABEL_MAP[pred_class]

    return pred_class, pred_label, pred_prob

pipeline = load_pipeline()
if pipeline is None:
    st.error("Model files not found! Place preprocessing.pkl and final_XGboost_model.pkl in the models/ directory.")
    st.stop()

xgb_model, scaler_img, pca, scaler_clinical = pipeline

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<div class="section-header">📷 Blood Smear Images</div>', unsafe_allow_html=True)
    uploaded_images = st.file_uploader(
        "Upload one or more peripheral blood smear images",
        type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
        help="Upload cell images from a single patient. Multiple images will be averaged.",
    )

    if uploaded_images:
        st.success(f"{len(uploaded_images)} image(s) uploaded")
        preview_cols = st.columns(min(len(uploaded_images), 5))
        for i, img_file in enumerate(uploaded_images[:5]):
            with preview_cols[i]:
                st.image(img_file, use_container_width=True, caption=f"Image {i+1}")
        if len(uploaded_images) > 5:
            st.caption(f"... and {len(uploaded_images) - 5} more")

with col_right:
    st.markdown('<div class="section-header">📋 Clinical Data</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ About Clinical Features", expanded=False):
        st.markdown(
            "Enter the peripheral blood differential count values as percentages. "
            "These represent the proportions of different white blood cell types."
        )

    demo1, demo2, demo3 = st.columns(3)
    with demo1:
        sex = st.selectbox("Sex", ["Female", "Male"], index=1)
        sex_val = 1 if sex == "Female" else 2
    with demo2:
        age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=45.0, step=0.1)
    with demo3:
        leucocytes = st.number_input("Leucocytes (×10³/µL)", min_value=0.0, max_value=1000.0, value=10.0, step=0.1)

    st.markdown("**Peripheral Blood Differential (%)**")

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        pb_myeloblast = st.number_input("Myeloblast", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    with r1c2:
        pb_promyelocyte = st.number_input("Promyelocyte", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    with r1c3:
        pb_myelocyte = st.number_input("Myelocyte", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    with r1c4:
        pb_metamyelocyte = st.number_input("Metamyelocyte", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        pb_neutrophil_band = st.number_input("Neutrophil Band", min_value=0.0, max_value=100.0, value=3.0, step=1.0)
    with r2c2:
        pb_neutrophil_seg = st.number_input("Neutrophil Seg.", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
    with r2c3:
        pb_eosinophil = st.number_input("Eosinophil", min_value=0.0, max_value=100.0, value=3.0, step=1.0)
    with r2c4:
        pb_basophil = st.number_input("Basophil", min_value=0.0, max_value=100.0, value=1.0, step=1.0)

    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        pb_monocyte = st.number_input("Monocyte", min_value=0.0, max_value=100.0, value=6.0, step=1.0)
    with r3c2:
        pb_lymph_typ = st.number_input("Lymphocyte (typical)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    with r3c3:
        pb_lymph_atyp = st.number_input("Atypical Lymph.", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

    pb_other = st.number_input("Other cells", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

clinical_data = {
    "sex_1f_2m": sex_val, "age": age, "leucocytes_per_ul": leucocytes,
    "pb_myeloblast": pb_myeloblast, "pb_promyelocyte": pb_promyelocyte,
    "pb_myelocyte": pb_myelocyte, "pb_metamyelocyte": pb_metamyelocyte,
    "pb_neutrophil_band": pb_neutrophil_band,
    "pb_neutrophil_segmented": pb_neutrophil_seg,
    "pb_eosinophil": pb_eosinophil, "pb_basophil": pb_basophil,
    "pb_monocyte": pb_monocyte, "pb_lymph_typ": pb_lymph_typ,
    "pb_lymph_atyp_react": pb_lymph_atyp, "pb_other": pb_other,
}

st.markdown("---")

if not uploaded_images:
    st.info("⬆️ Upload at least one blood smear image to enable prediction.")

if st.button("🚀 Run Prediction", type="primary", use_container_width=True, disabled=not uploaded_images):
    with st.spinner("Loading ResNet50 feature extractor..."):
        base_model = load_resnet()

    with st.spinner(f"Extracting features from {len(uploaded_images)} image(s)..."):
        image_features = extract_image_features(uploaded_images, base_model)

    with st.spinner("Running multimodal prediction..."):
        pred_class, pred_label, pred_prob = predict_patient(
            image_features, clinical_data,
            xgb_model, scaler_img, pca, scaler_clinical,
        )

        st.session_state.last_prediction = {
        "label": pred_label,
        "confidence": round(pred_prob[pred_class] * 100, 2),
        "probabilities": {LABEL_MAP[i]: round(pred_prob[i] * 100, 2) for i in range(5)},
        "clinical": clinical_data,
        "images_count": len(uploaded_images),
    }

    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    color = LABEL_COLORS.get(pred_label, "#E63946")
    confidence = pred_prob[pred_class] * 100

    st.markdown(
        f'<div class="result-card" style="border-color: {color};">'
        f'<div class="result-label" style="color: {color};">{pred_label}</div>'
        f'<div class="result-conf">Confidence: {confidence:.1f}%</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="info-card"><p>'
        f'<strong style="color: #FFF;">{pred_label}:</strong> '
        f'{LABEL_DESCRIPTIONS.get(pred_label, "")}</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Class Probabilities**")
    label_names = list(LABEL_MAP.values())
    probs = [(label_names[i], pred_prob[i]) for i in range(5)]
    probs.sort(key=lambda x: x[1], reverse=True)

    for cls_name, prob_val in probs:
        bar_color = LABEL_COLORS.get(cls_name, "#888")
        marker = " ◀" if cls_name == pred_label else ""
        st.markdown(
            f'<div class="prob-row">'
            f'<div class="prob-header">'
            f'<span class="prob-name">{cls_name}{marker}</span>'
            f'<span class="prob-val">{prob_val*100:.2f}%</span>'
            f'</div>'
            f'<div class="prob-bg">'
            f'<div class="prob-fill" style="width: {prob_val*100:.1f}%; background: {bar_color};"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with st.expander("📋 Clinical Data Summary"):
        summary_df = pd.DataFrame([clinical_data]).T
        summary_df.columns = ["Value"]
        summary_df.index.name = "Feature"
        st.dataframe(summary_df, use_container_width=True)

    
    st.markdown("---")
    st.info("💡 Have questions about your results? Go to the **AI Chatbot** page ")

    st.warning(
        "⚠️ This tool is for research and educational purposes only. "
        "Not a substitute for professional medical diagnosis."
    )