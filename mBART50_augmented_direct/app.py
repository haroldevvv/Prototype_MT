import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================================
# Load Model Function
# ==========================================================
@st.cache_resource
def load_model():
    # Hugging Face Hub model ID
    model_path = "haroldevvv/my-mbart50-translation-model"

    # Load tokenizer & model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Auto-select device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return tokenizer, model, device


# ==========================================================
# Initialize model and tokenizer
# ==========================================================
tokenizer, model, device = load_model()

# ==========================================================
# Streamlit Interface
# ==========================================================
st.title("🌐 Rinconada → English Translator (mBART50)")

st.markdown(
    """
    This prototype uses a fine-tuned **mBART50** model to translate text  
    from **Rinconada** to **English**.
    """
)

# Input text area
rin_text = st.text_area("Enter Rinconada text:", height=150)

# Translate button
if st.button("Translate"):
    if rin_text.strip() == "":
        st.warning("⚠️ Please enter some text to translate.")
    else:
        with st.spinner("Translating... Please wait."):
            # Tokenize input
            inputs = tokenizer(rin_text, return_tensors="pt", truncation=True, padding=True).to(device)

            # Generate translation
            outputs = model.generate(**inputs, max_length=200)

            # Decode output
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Display result
            st.subheader("📝 English Translation:")
            st.success(translation)
