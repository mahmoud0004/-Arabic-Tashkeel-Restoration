import re
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
import streamlit as st


# =========================
# Paths
# =========================
PROJECT_DIR = Path.cwd()
MODEL_DIR = PROJECT_DIR / "models"

BEST_MODEL_PATH = MODEL_DIR / "best_tashkeel_model.pt"
ARTIFACTS_PATH = MODEL_DIR / "tashkeel_artifacts.pkl"

PAD_CHAR_ID = 0


# =========================
# Text Processing
# =========================
DIACRITICS_PATTERN = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")
ARABIC_KEEP_PATTERN = re.compile(r"[^\u0600-\u06FF\u064B-\u0652\u0670\s\.,،؛؟!]")
SPACE_PATTERN = re.compile(r"\s+")


def clean_line(line: str) -> str:
    line = ARABIC_KEEP_PATTERN.sub(" ", line.strip())
    line = SPACE_PATTERN.sub(" ", line)
    return line.strip()


def remove_tashkeel(text: str) -> str:
    return DIACRITICS_PATTERN.sub("", text)


def prepare_input_text(text):
    return remove_tashkeel(clean_line(text))


# =========================
# Model
# =========================
class BiLSTMTashkeelModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        embedding_dim=128,
        hidden_dim=256,
        dropout=0.25,
        use_attention=False
    ):
        super().__init__()

        self.use_attention = use_attention

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=PAD_CHAR_ID
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        model_dim = hidden_dim * 2

        if use_attention:
            self.attention = nn.MultiheadAttention(
                model_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(model_dim, num_labels)

    def forward(self, x):
        pad_mask = x.eq(PAD_CHAR_ID)

        x = self.embedding(x)
        x, _ = self.lstm(x)

        if self.use_attention:
            attn_out, _ = self.attention(
                x,
                x,
                x,
                key_padding_mask=pad_mask,
                need_weights=False
            )
            x = self.norm(x + attn_out)

        x = self.dropout(x)
        return self.fc(x)


# =========================
# Load Model
# =========================
@st.cache_resource
def load_model_and_artifacts():
    if not ARTIFACTS_PATH.exists():
        st.error("❌ artifacts file not found: models/tashkeel_artifacts.pkl")
        st.stop()

    if not BEST_MODEL_PATH.exists():
        st.error("❌ model file not found: models/best_tashkeel_model.pt")
        st.stop()

    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BiLSTMTashkeelModel(
        vocab_size=artifacts["vocab_size"],
        num_labels=artifacts["num_labels"],
        embedding_dim=artifacts["embedding_dim"],
        hidden_dim=artifacts["hidden_dim"],
        dropout=artifacts["dropout"],
        use_attention=artifacts["use_attention"]
    ).to(device)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    return model, artifacts, device


model, artifacts, device = load_model_and_artifacts()

char2idx = artifacts["char2idx"]
idx2label = artifacts["idx2label"]
MAX_LEN = artifacts["max_len"]


# =========================
# Prediction
# =========================
def predict_tashkeel(text, return_confidence=False):
    cleaned = prepare_input_text(text)

    if not cleaned:
        return ("", 0.0) if return_confidence else ""

    pieces = []
    confidences = []

    for start in range(0, len(cleaned), MAX_LEN):
        chunk = cleaned[start:start + MAX_LEN]

        ids = [char2idx.get(ch, PAD_CHAR_ID) for ch in chunk]
        x = torch.tensor([ids], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)

            pred_ids = torch.argmax(probs, dim=-1)[0].cpu().numpy().tolist()
            conf = torch.max(probs, dim=-1).values[0].cpu().numpy().tolist()

        for ch, pred_id, c in zip(chunk, pred_ids, conf):
            pieces.append(ch)
            pieces.append(idx2label.get(int(pred_id), ""))
            confidences.append(float(c))

    output = "".join(pieces)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    return (output, avg_conf) if return_confidence else output


# =========================
# Streamlit Page
# =========================
st.set_page_config(
    page_title="Arabic Tashkeel AI",
    page_icon="🕌",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
}

.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.hero {
    text-align: center;
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(135deg, #111827, #1f2937);
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    animation: fadeIn 1s ease-in-out;
}

.hero h1 {
    font-size: 46px;
    color: #f8fafc;
    margin-bottom: 10px;
}

.hero p {
    font-size: 20px;
    color: #cbd5e1;
}

.card {
    padding: 25px;
    border-radius: 22px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 10px 35px rgba(0,0,0,0.25);
    animation: slideUp 0.8s ease-in-out;
}

.output-box {
    direction: rtl;
    text-align: right;
    font-size: 28px;
    line-height: 2.1;
    padding: 25px;
    border-radius: 20px;
    background: #f8fafc;
    color: #111827;
    border-right: 8px solid #38bdf8;
}

.metric-box {
    text-align: center;
    padding: 18px;
    border-radius: 18px;
    background: linear-gradient(135deg, #0284c7, #0ea5e9);
    color: white;
    font-size: 22px;
    font-weight: 700;
}

@keyframes fadeIn {
    from {opacity: 0; transform: scale(0.97);}
    to {opacity: 1; transform: scale(1);}
}

@keyframes slideUp {
    from {opacity: 0; transform: translateY(30px);}
    to {opacity: 1; transform: translateY(0);}
}

.stTextArea textarea {
    direction: rtl;
    text-align: right;
    font-size: 22px;
    border-radius: 18px;
}

.stButton button {
    width: 100%;
    border-radius: 18px;
    height: 55px;
    font-size: 20px;
    font-weight: 700;
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: white;
    border: none;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 25px rgba(14,165,233,0.4);
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>Arabic Tashkeel Restoration AI</h1>
    <p>BiLSTM Model for restoring Arabic diacritics automatically</p>
</div>
""", unsafe_allow_html=True)

st.write("")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Enter Arabic Text")

    text = st.text_area(
        "Text without tashkeel",
        value="ذهب الطالب الى المدرسة",
        height=180,
        label_visibility="collapsed"
    )

    run_btn = st.button("Restore Tashkeel")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Quick Examples")

    examples = [
        "بسم الله الرحمن الرحيم",
        "الحمد لله رب العالمين",
        "اللغة العربية لغة جميلة",
        "ذهب الطالب الى المدرسة",
        "قرأ المعلم الدرس"
    ]

    selected_example = st.selectbox("Choose example", examples)

    if st.button("Use Example"):
        text = selected_example
        run_btn = True

    st.markdown("</div>", unsafe_allow_html=True)


st.write("")

if run_btn:
    with st.spinner("Restoring tashkeel..."):
        output, confidence = predict_tashkeel(text, return_confidence=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Result")

    st.markdown(f"""
    <div class="output-box">
        {output}
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-box">
            Confidence<br>{confidence * 100:.2f}%
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-box">
            Characters<br>{len(prepare_input_text(text))}
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-box">
            Device<br>{device.upper()}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


with st.expander("Project Info"):
    st.write("Model: BiLSTM + Optional Attention")
    st.write("This app loads the trained model from `models/best_tashkeel_model.pt`")
    st.write("It also loads preprocessing artifacts from `models/tashkeel_artifacts.pkl`")