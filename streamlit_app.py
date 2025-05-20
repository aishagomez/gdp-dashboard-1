import streamlit as st
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F

# Cargar modelo y tokenizer
@st.cache_resource
def load_model():
    model = torch.load("beto_classifier_complete.pt", map_location=torch.device("cpu"))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    return model, tokenizer

model, tokenizer = load_model()

# Interfaz Streamlit
st.title("📰 Clasificador de Noticias - BETO")
st.write("Introduce una noticia en español y el modelo dirá si es **falsa** o **real**.")

text_input = st.text_area("✍️ Escribe aquí tu noticia...", height=200)

if st.button("Clasificar"):
    if text_input.strip() == "":
        st.warning("Por favor, escribe una noticia.")
    else:
        # Tokenización
        inputs = tokenizer(
            text_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        labels = ["❌ Falsa", "✅ Real"]
        st.subheader(f"Resultado: {labels[pred]}")
        st.write(f"Confianza: {probs[0][pred]*100:.2f}%")
