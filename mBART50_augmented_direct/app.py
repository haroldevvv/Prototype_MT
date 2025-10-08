import streamlit as st
import torch
import time
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# ============================
#  Page Config
# ============================
st.set_page_config(
    page_title="Rinconada → English Translator",
    page_icon="🌐",
    layout="centered"
)

# ============================
#  Model Loader (cached)
# ============================
@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "haroldevvv/my-mbart50-translation-model"
    try:
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f" Failed to load model: {e}")
        st.stop()

# ============================
#  UI Header
# ============================
st.title("🌐 Rinconada → English Translator (mBART50)")
st.markdown(
    "Translate Rinconada text into English using a fine-tuned **mBART50** model."
)

# ============================
#  Load Model
# ============================
tokenizer, model = load_model()

# ============================
#  Input Section
# ============================
text = st.text_area("Enter Rinconada text:")

# ============================
#  Translation Logic
# ============================
if st.button("Translate"):
    if not text.strip():
        st.warning(" Please enter some text before translating.")
    else:
        with st.spinner("Translating... please wait"):
            start_time = time.time()
            try:
                forced_bos = model.config.forced_bos_token_id  # en_XX from training
                inputs = tokenizer(text, return_tensors="pt")

                if torch.cuda.is_available():
                    model.to("cuda")
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_length=200,
                    num_beams=5,
                    early_stopping=True
                )

                translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
                elapsed = time.time() - start_time

                st.success(f" Translation complete! ({elapsed:.2f}s)")
                st.text_area("English Translation:", value=translation, height=150)

            except Exception as e:
                st.error(f" Translation failed: {e}")
