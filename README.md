# Rinconada → English Translator (mBART50)

This prototype is a **web-based machine translation application** built with **Streamlit** and a **fine-tuned mBART50 model** from Hugging Face.  
It translates text from **Rinconada** to **English** using a custom-trained multilingual neural machine translation model.

---

##  Features

- 🌐 Rinconada → English translation
-  Fine-tuned **mBART50** model hosted on Hugging Face
-  GPU acceleration (automatically used if available)
-  Cache clearing and model unloading to manage memory
-  Simple, interactive Streamlit UI

---

##  Tech Stack

- **Python 3.8+**
- **Streamlit**
- **PyTorch**
- **Hugging Face Transformers**
- **mBART50 (MBartForConditionalGeneration)**

---

##  Prerequisites

Make sure you have the following installed:

- Python **3.8 or newer**
- pip
- (Optional but recommended) NVIDIA GPU with CUDA support

---

##  Installation

### 1. Clone the repository

```
git clone https://github.com/haroldevvv/Prototype_MT.git
cd Prototype_MT/mBART50_augmented_direct
```
### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

### 3. Activate it 
For Windows:
```venv\Scripts\activate```
For macOS/Linux:
```source venv/bin/activate```

### 4. Install dependencies

```pip install streamlit torch transformers sentencepiece```

### 5. Run the application 

```streamlit run app.py```

### 6. Open it in your localhost 

