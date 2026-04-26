import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

model_path = "deployment/emotion_model"

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

model.eval()

id2label = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "sadness",
    4: "neutral"
}

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    return id2label[pred], float(probs[0][pred])

# Test
if __name__ == "__main__":
    text = input("Enter text: ")
    emotion, confidence = predict_emotion(text)
    print("Emotion:", emotion)
    print("Confidence:", round(confidence, 2))