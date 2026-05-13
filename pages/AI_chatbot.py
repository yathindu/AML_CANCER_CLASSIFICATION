import streamlit as st
import requests

st.set_page_config(page_title="AI Chatbot — AML Detection", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 900px; }
    .chat-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 1rem 0;
        border-bottom: 1px solid #2A2E35;
        margin-bottom: 1rem;
    }
    .chat-icon {
        width: 42px;
        height: 42px;
        background: rgba(230, 57, 70, 0.12);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }
    .chat-title { font-size: 1.2rem; font-weight: 700; color: #FAFAFA; }
    .chat-sub { font-size: 0.82rem; color: #888; }
    .welcome-box {
        text-align: center;
        padding: 3rem 1rem;
    }
    .welcome-box h3 { font-weight: 700; margin-bottom: 0.5rem; }
    .welcome-box p { color: #A8DADC; font-size: 0.92rem; line-height: 1.6; }
    .quick-btn {
        display: block;
        width: 100%;
        text-align: left;
        background: transparent;
        color: #A8DADC;
        border: 1px solid #2A2E35;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.82rem;
        cursor: pointer;
        transition: border-color 0.2s;
    }
    .quick-btn:hover { border-color: #E63946; }
</style>
""", unsafe_allow_html=True)

SYSTEM_PROMPT = """You are an expert AI assistant specialized in Acute Myeloid Leukemia (AML) 
and hematological oncology. You are part of an AML Leukemia Detection System that uses 
multimodal AI (ResNet50 image features + clinical data + XGBoost) to classify AML subtypes.

You can answer questions about:
- AML subtypes: Control, NPM1, PML_RARA, RUNX1_RUNX1T1, CBFB_MYH11
- Blood cell morphology and cytomorphology analysis
- Peripheral blood differential counts and their significance
- AML diagnosis, prognosis, and treatment approaches
- The technical pipeline of this detection system

Key facts about the system:
- Uses the TCIA AML-Cytomorphology dataset (189 patients, 81,214 images)
- Image branch: ResNet50 (pretrained ImageNet) extracts 2048-dim features
- Clinical branch: 15 features from peripheral blood differential counts
- Features are fused (PCA + concat) and classified by tuned XGBoost
- 5 output classes: Control, NPM1, PML_RARA, RUNX1_RUNX1T1, CBFB_MYH11
- Achieved 84.21% accuracy and 80.72% macro F1 on the test set

Always include a disclaimer that your responses are for educational purposes only."""

def call_llm(messages, api_key):
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"
    
with st.sidebar:
    st.markdown("### ⚙️ Chatbot Settings")
    st.markdown("---")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com/keys",
    )

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Quick Questions**")
    quick_questions = [
        "What is AML?",
        "Explain the NPM1 subtype",
        "How does ResNet50 extract features?",
        "What do blood differential counts mean?",
        "What is PML-RARA (APL)?",
        "How does this detection system work?",
    ]
    for q in quick_questions:
        if st.button(q, key=f"qq_{q}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

st.markdown("""
<div class="chat-header">
    <div class="chat-icon">🤖</div>
    <div>
        <div class="chat-title">AML AI Assistant</div>
        <div class="chat-sub">Ask about AML, morphology, diagnosis, or this detection system</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not api_key:
    st.markdown("""
    <div class="welcome-box">
        <div style="font-size: 2.5rem; margin-bottom: 1rem;">🧬</div>
        <h3>Welcome to AML AI Assistant</h3>
        <p>Enter your Groq API key in the sidebar to start chatting.<br>
        Get a free key at <a href="https://console.groq.com/keys" target="_blank">console.groq.com/keys</a></p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not st.session_state.chat_history:
    st.markdown("""
    <div class="welcome-box">
        <div style="font-size: 2.5rem; margin-bottom: 1rem;">🧬</div>
        <h3>Welcome to AML AI Assistant</h3>
        <p>Ask me anything about AML subtypes, blood cell morphology,<br>
        diagnosis, or how this detection system works.</p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about AML...")

if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    system = SYSTEM_PROMPT

    if "last_prediction" in st.session_state:
        pred = st.session_state.last_prediction
        system += f"""

The user just ran a prediction with the following results:
- Predicted class: {pred['label']}
- Confidence: {pred['confidence']}%
- All probabilities: {pred['probabilities']}
- Clinical data used: {pred['clinical']}
- Number of images analyzed: {pred['images_count']}

Use this context to answer their questions about the results. Explain what 
the prediction means, suggest next steps, and discuss the specific subtype 
if they ask."""

    messages = [{"role": "system", "content": system}]
    for msg in st.session_state.chat_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = call_llm(messages, api_key)
        st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})