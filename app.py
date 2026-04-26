import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# Load model
model_path = "deployment/emotion_model"

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

model.eval()

# Label mapping
id2label = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "sadness",
    4: "neutral"
}

# Colors
emotion_colors = {
    "joy": "#FFD700",
    "sadness": "#1E90FF",
    "anger": "#FF4B4B",
    "fear": "#8A2BE2",
    "neutral": "#A9A9A9"
}

# Emojis
emoji_map = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "neutral": "😐"
}

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0].tolist()
    pred = probs.index(max(probs))

    return id2label[pred], probs


# ----------- UI -------------

st.set_page_config(page_title="Emotion Detector", page_icon="🎭", layout="centered")

# Custom CSS (Animation + Styling)
st.markdown("""
<style>
.big-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #4CAF50;
    animation: fadeIn 2s;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🎭 Emotion Detection System</div>", unsafe_allow_html=True)

st.write("### Enter text to analyze emotion:")

text = st.text_area("", height=150)

if st.button("🔍 Predict Emotion"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        emotion, probs = predict_emotion(text)

        color = emotion_colors[emotion]
        emoji = emoji_map[emotion]

        # 🎨 Colored result box
        st.markdown(
            f"""
            <div style='
                background-color:{color};
                padding:20px;
                border-radius:15px;
                text-align:center;
                color:white;
                font-size:28px;
                font-weight:bold;
                animation: fadeIn 1.5s;
            '>
                {emoji} {emotion.upper()}
            </div>
            """,
            unsafe_allow_html=True
        )

        # 📊 Confidence chart
        df = pd.DataFrame({
            "Emotion": list(id2label.values()),
            "Confidence": probs
        })

        st.write("### 📊 Emotion Confidence Distribution")
        st.bar_chart(df.set_index("Emotion"))

        # 📈 Progress bars
        st.write("### 📈 Detailed Confidence")
        for emo, prob in zip(id2label.values(), probs):
            st.write(f"{emo} ({prob:.2f})")
            st.progress(int(prob * 100))