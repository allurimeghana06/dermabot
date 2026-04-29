import os
import streamlit as st
import torch
import torch.nn.functional as F
import timm  # <-- use timm, not torchvision (training used timm)
from torchvision import transforms
from PIL import Image
from google import genai

# --- PAGE CONFIG ---
st.set_page_config(page_title="DERMABOT", page_icon="🩺", layout="wide")

# --- CUSTOM THEME (CSS) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    [data-testid="stChatMessage"] {
        border-radius: 20px;
        background-color: #1E2129;
        border: 1px solid #30363D;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #4CAF50;
    }
    .main-title {
        text-align: center;
        color: #4CAF50;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .main-subtitle {
        text-align: center;
        color: #8B949E;
        font-size: 1.1rem;
        margin-top: 0;
    }
    </style>
    <h1 class='main-title'>🧴 DERMABOT</h1>
    <p class='main-subtitle'>AI-powered Skin Disease Assistant</p>
    """, unsafe_allow_html=True)

# =========================
# 🔑 API SETUP
# =========================
# SECURITY: Store the API key in Streamlit secrets, NOT in the code.
# Create a file `.streamlit/secrets.toml` with:
#     GEMINI_API_KEY = "your_actual_key_here"
# Then the line below reads it safely.
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY not set. Add it to .streamlit/secrets.toml or as an environment variable.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# =========================
# 🧠 MODEL SETUP — MUST match training exactly
# =========================
MODEL_PATH = "best_model.pth"        # rename your file to this (or update the path)
CLASS_NAMES_PATH = "class_names.txt"
IMG_SIZE = 224

@st.cache_resource
def load_model():
    """Load the trained model checkpoint."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()

    # Load checkpoint with metadata (model_name, classes, etc.)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    # Rebuild architecture using metadata from the checkpoint
    model = timm.create_model(
        checkpoint['model_name'],          # e.g. 'convnext_small.fb_in22k_ft_in1k'
        pretrained=False,                   # we'll load OUR weights, not ImageNet
        num_classes=checkpoint['num_classes'],
    )

    # strict=True — this should NOT silently ignore any weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    # Pull class names directly from the checkpoint (guaranteed correct order)
    class_names = checkpoint['classes']
    img_size = checkpoint.get('img_size', 224)

    return model, class_names, img_size

# Load model + classes
model, CLASS_NAMES, IMG_SIZE = load_model()

# Pretty display names (shown to users) — but predictions still come from CLASS_NAMES
DISPLAY_NAMES = {
    "Acne":               "Acne",
    "Actinic_Keratosis":  "Actinic Keratosis",
    "Benign_tumors":      "Benign Tumors",
    "Bullous":            "Bullous Disease",
    "DrugEruption":       "Drug Eruption",
    "Eczema":             "Eczema",
    "Infestations_Bites": "Infestations / Bites",
    "Lichen":             "Lichen",
    "Psoriasis":          "Psoriasis",
    "Rosacea":            "Rosacea",
    "Seborrh_Keratoses":  "Seborrheic Keratoses",
    "SkinCancer":         "Skin Cancer",
    "Tinea":              "Tinea (Ringworm)",
    "Unknown_Normal":     "Unknown / Normal Skin",
    "Vascular_Tumors":    "Vascular Tumors",
    "Vasculitis":         "Vasculitis",
    "Vitiligo":           "Vitiligo",
    "Warts":              "Warts",
}

# === Inference transform — MUST match training's val_tf exactly ===
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # <-- THIS WAS MISSING
])


@torch.no_grad()
def predict(pil_image, top_k=3, use_tta=True):
    """
    Run inference. Returns list of (display_name, raw_class_name, probability).
    use_tta=True averages over 4 image flips for ~+1-2% accuracy.
    """
    img = pil_image.convert("RGB")
    x = transform(img).unsqueeze(0)

    if use_tta:
        views = [
            x,
            torch.flip(x, dims=[3]),    # h-flip
            torch.flip(x, dims=[2]),    # v-flip
            torch.flip(x, dims=[2, 3]), # 180°
        ]
        probs = sum(F.softmax(model(v), dim=1) for v in views) / len(views)
    else:
        probs = F.softmax(model(x), dim=1)

    probs = probs.squeeze(0).numpy()
    top_idx = probs.argsort()[::-1][:top_k]

    return [
        (DISPLAY_NAMES.get(CLASS_NAMES[i], CLASS_NAMES[i]), CLASS_NAMES[i], float(probs[i]))
        for i in top_idx
    ]


# =========================
# 🤖 AI LOGIC
# =========================
def get_ai_response(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error connecting to AI Assistant: {str(e)}"


# =========================
# 💬 CHAT INTERFACE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "👋 **Hello! I am your AI Skin Assistant.**\n\nPlease upload a clear photo of the skin area in the sidebar to begin analysis."}
    ]

# Sidebar
with st.sidebar:
    st.title("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "png", "jpeg"])
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.pop("last_processed", None)
        st.rerun()

    st.info("**Disclaimer:** This is an AI tool for educational purposes only. Always consult a certified dermatologist for medical diagnosis.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg:
            st.image(msg["image"], width=400)

# Handle new upload
# Handle new upload OR file removal
# Handle file removal
# Handle file removal — clear everything when X is clicked
if uploaded_file is None:
    if st.session_state.get("last_processed") is not None:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "👋 **Hello! I am your AI Skin Assistant.**\n\nPlease upload a clear photo of the skin area in the sidebar to begin analysis."}
        ]
        st.session_state.last_processed = None
        st.rerun()
else:
    file_id = f"file_{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("last_processed") != file_id:
        image = Image.open(uploaded_file)

        # User Message
        st.session_state.messages.append({
            "role": "user",
            "content": "I've uploaded an image for analysis.",
            "image": image,
        })

        # Assistant Processing
        with st.chat_message("assistant"):
            with st.spinner("Analyzing skin patterns..."):
                top_predictions = predict(image, top_k=3, use_tta=True)

                top_display, top_raw, top_conf = top_predictions[0]
                top_conf_pct = top_conf * 100

            # Show top-1 prominently
            st.markdown("### Analysis Complete")
            st.metric(
                label="Most likely",
                value=top_display,
                delta=f"{top_conf_pct:.1f}% confidence",
            )

            # Show top-3 with progress bars
            st.markdown("**Top 3 predictions:**")
            for display, raw, prob in top_predictions:
                st.write(f"**{display}**")
                st.progress(prob, text=f"{prob*100:.1f}%")

            # Warn on low confidence
            if top_conf < 0.5:
                st.warning(
                    "⚠️ Low confidence prediction. The image may be unclear or show a "
                    "condition outside the model's training set. Please consult a dermatologist."
                )

            # Special handling for Unknown_Normal
            if top_raw == "Unknown_Normal":
                st.info(
                    "ℹ️ The model thinks this is normal skin or doesn't recognize a clear "
                    "disease pattern. If you're concerned about a specific lesion, consider "
                    "consulting a dermatologist."
                )

            # Get AI explanation
            with st.spinner("Getting AI explanation..."):
                prompt = (
                    f"Briefly explain the skin condition '{top_display}' in 2-3 short paragraphs. "
                    f"Cover: what it is, common symptoms, and whether it typically requires medical attention. "
                    f"Use simple, supportive language. Do not give medical advice — recommend seeing a dermatologist for diagnosis."
                )
                explanation = get_ai_response(prompt)

            st.markdown("---")
            st.markdown(explanation)

            # Save to history
            top3_text = "\n".join(
                f"- **{d}**: {p*100:.1f}%" for d, _, p in top_predictions
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"### Analysis Result\n"
                    f"**Most likely:** {top_display} ({top_conf_pct:.1f}% confidence)\n\n"
                    f"**Top 3:**\n{top3_text}\n\n"
                    f"---\n\n{explanation}"
                ),
            })

        st.session_state.last_processed = file_id
        st.rerun()

# Text chat input
if user_input := st.chat_input("Ask me a question about the results..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Use last assistant message as context
            recent_assistant = next(
                (m["content"] for m in reversed(st.session_state.messages[:-1])
                 if m["role"] == "assistant"),
                "",
            )
            final_prompt = (
                f"Previous conversation context:\n{recent_assistant}\n\n"
                f"User question: {user_input}\n\n"
                f"Answer as a helpful, professional medical assistant. "
                f"Always remind users to consult a dermatologist for diagnosis."
            )
            reply = get_ai_response(final_prompt)
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
