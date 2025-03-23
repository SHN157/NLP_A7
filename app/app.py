import streamlit as st
import torch
from transformers import BertForSequenceClassification, AutoTokenizer, BertConfig

# Set title
st.title("Toxic Comment Classifier")
st.write("Type a comment below to check if it's toxic or not.")

# Input box
user_input = st.text_area("Enter your comment here:")

# Load model and tokenizer (Even-Layer Student)
@st.cache_resource
def load_model():
    path = "/Users/soehtetnaing/Documents/GitHub/NLP_A7/even_layer_student"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)

    # Load 6-layer distilled model config
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.num_hidden_layers = 6
    config.num_labels = 2

    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load(f"{path}/model_weights.pt", map_location="cpu"))
    model.eval()
    
    return model, tokenizer

model, tokenizer = load_model()

# Predict
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return pred, confidence

# Run prediction
if st.button("🚀 Classify"):
    if user_input.strip():
        label, confidence = predict(user_input)
        label_name = " Non-Toxic" if label == 0 else " Toxic"
        st.markdown(f"### Prediction: {label_name}")
        st.markdown(f"**Confidence:** {confidence:.2%}")
    else:
        st.warning("Please enter some text.")
