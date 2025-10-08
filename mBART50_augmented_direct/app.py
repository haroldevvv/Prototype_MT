import streamlit as st
from transformers import pipeline

# ==========================================================
# Load Model Function (with src/tgt language setup)
# ==========================================================
@st.cache_resource
def load_model():
    """
    Load the translation pipeline from Hugging Face with language codes.
    """
    translator = pipeline(
        "translation",
        model="haroldevvv/my-mbart50-translation-model",
        src_lang="rin_Latn",  
        tgt_lang="en_XX"       # Target = English
    )
    return translator


# ==========================================================
# Initialize the Translator
# ==========================================================
translator = load_model()


# ==========================================================
# Streamlit Interface
# ==========================================================
st.title("🌐 Rinconada → English Translator (mBART50)")

st.markdown(
    """
    This prototype uses a fine-tuned **mBART50** model to translate text  
    from **Rinconada** to **English**, powered by Hugging Face hosted inference.
    """
)

# Input text area
rin_text = st.text_area("Enter Rinconada text:", height=150)

# Translate button
if st.button("Translate"):
    if rin_text.strip() == "":
        st.warning(" Please enter some text to translate.")
    else:
        with st.spinner("Translating... Please wait."):
            try:
                result = translator(rin_text, max_length=200)
                translation = result[0]["translation_text"]

                st.subheader("📝 English Translation:")
                st.success(translation)
            except Exception as e:
                st.error(f" Translation failed: {str(e)}")
