import streamlit as st
import torch
import time
import gc
import psutil
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
#  Sidebar: Maintenance & Memory Monitor
# ============================

def smart_reset(threshold_mb: int = 1500):
    """
    Smart memory management button.
    - If memory > threshold_mb: clear cache & restart.
    - Else: unload model only.
    """
    process = psutil.Process()
    mem_usage_mb = process.memory_info().rss / 1024 / 1024  # MB

    st.sidebar.subheader(" Resource Monitor")
    st.sidebar.write(f" RAM Usage: {mem_usage_mb:.0f} MB")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
        st.sidebar.write(f" GPU Memory: {gpu_mem:.0f} MB")

    if st.sidebar.button(" Smart Reset"):
        if mem_usage_mb > threshold_mb:
            st.sidebar.warning("High memory usage detected — clearing cache and restarting...")
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
        else:
            try:
                del model
                del tokenizer
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            st.sidebar.success("Model unloaded. Cache kept for faster reloads.")


# ============================
#  Model Loader (cached)
# ============================
@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "haroldevvv/my-mbart50-translation-model"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model


# ============================
#  UI Header
# ============================
st.title("🌐 Rinconada → English Translator (mBART50)")
st.markdown("Translate Rinconada text into English using a fine-tuned **mBART50** model.")


# ============================
#  Load Model
# ============================
try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f" Failed to load model: {e}")
    st.stop()


# ============================
#  Input Section
# ============================
text = st.text_area("Enter Rinconada text:")


# ============================
#  Translation Logic
# ============================
if st.button("Translate"):
    if not text.strip():
        st.warning("Please enter some text before translating.")
    else:
        with st.spinner("Translating... please wait"):
            start_time = time.time()
            try:
                forced_bos = model.config.forced_bos_token_id
                inputs = tokenizer(text, return_tensors="pt")

                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                with torch.no_grad():
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


# ============================
#  Sidebar Smart Reset
# ============================
st.sidebar.header("Maintenance")
smart_reset(threshold_mb=1500)
