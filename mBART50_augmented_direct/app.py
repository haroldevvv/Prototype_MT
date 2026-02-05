import streamlit as st
import torch
import time
import gc
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

st.set_page_config(
    page_title="Rinconada → English Translator",
    page_icon="🌐",
    layout="centered"
)

TGT_LANG = "en_XX"

if st.sidebar.button("Clear Cache and Restart"):
    st.cache_resource.clear()
    st.cache_data.clear()
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "haroldevvv/my-mbart50-translation-model"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()

    if torch.cuda.is_available():
        model.to("cuda")

    return tokenizer, model

st.title("🌐 Rinconada → English Translator")
st.markdown("Translate Rinconada text into English using a fine-tuned **mBART50** model.")

tokenizer, model = load_model()

tokenizer.tgt_lang = TGT_LANG

text = st.text_area("Enter Rinconada text:")

if st.button("Translate"):
    if not text.strip():
        st.warning("Please enter some text before translating.")
    else:
        with st.spinner("Translating... please wait"):
            start_time = time.time()
            try:
                inputs = tokenizer(text, return_tensors="pt")

                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.lang_code_to_id[TGT_LANG],
                        max_length=128,
                        num_beams=5,
                        early_stopping=True
                    )

                translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
                elapsed = time.time() - start_time

                st.success(f"Translation complete! ({elapsed:.2f}s)")
                st.text_area("English Translation:", value=translation, height=150)

            except Exception as e:
                st.error(f"Translation failed: {e}")

if st.sidebar.button("Unload Model (Free Memory)"):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.success("Model unloaded. Refresh to reload.")
