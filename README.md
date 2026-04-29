# DERMABOT — AI Skin Disease Classifier

A Streamlit web app that classifies 18 skin disease categories using a 
PyTorch ConvNeXt model, with AI-powered explanations via Google Gemini.

## Features
- 18-class skin disease classification (~80% accuracy with TTA)
- Top-3 predictions with confidence scores
- AI-generated explanations of each condition
- Chat interface for follow-up questions

## Setup

1. Clone the repo (Git LFS will download the model file automatically):
```bash
   git clone https://github.com/allurimeghana06/dermabot.git
   cd dermabot
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Add your Gemini API key to `.streamlit/secrets.toml`:
```toml
   GEMINI_API_KEY = "your_key_here"
```
   Get a free key at https://aistudio.google.com/apikey

4. Run the app:
```bash
   streamlit run app.py
```

## Disclaimer
This is an educational tool only. Always consult a certified dermatologist for medical diagnosis.

## Tech Stack
- PyTorch + timm (ConvNeXt-Small backbone)
- Streamlit (web UI)
- Google Gemini API (AI explanations)
