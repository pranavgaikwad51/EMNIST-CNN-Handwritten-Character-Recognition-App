import streamlit as st
import numpy as np
import pickle
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
from streamlit_drawable_canvas import st_canvas
import base64
import os

# -----------------------------------------------------------
# 0️⃣ Page Config & Setup
# -----------------------------------------------------------
st.set_page_config(
    page_title="EMNIST Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------
# 👤 Sidebar Developer Info (ADDED AS REQUESTED)
# -----------------------------------------------------------
with st.sidebar:
    st.markdown("## 👨‍💻 Developer Info")
    st.markdown("""
    **Pranav Gaikwad**  
    📧 Email: [gaikwadpranav988@gmail.com](mailto:gaikwadpranav988@gmail.com)  
    🔗 LinkedIn: [Click Here](https://www.linkedin.com/in/pranav-gaikwad-0b94032a)  
    🧠 GitHub: [Click Here](https://github.com/pranavgaikwad51)  
    📱 Phone: `7028719844`
    """)
    st.write("---")
    st.markdown("⭐ *Thanks for visiting!* 😊")


# -----------------------------------------------------------
# ✨ Enhanced Background & Styling
# -----------------------------------------------------------
@st.cache_resource
def get_base64_of_bin_file(bin_file):
    """Reads a binary file and returns its base64 encoded string."""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.error(f"Error reading background image: {e}")
        return None


def set_app_background(jpg_file):
    """Sets blurred app background."""
    if not os.path.exists(jpg_file):
        return
        
    bin_str = get_base64_of_bin_file(jpg_file)
    if bin_str is None:
        return
    
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: url("data:image/jpeg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        filter: blur(10px);
        opacity: 0.3;
        z-index: -1;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Apply background
set_app_background("alpha and digit.jpg")

# -----------------------------------------------------------
# 1️⃣ Load Model
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "cnn_emnist_digits_alphabets.pkl"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_model()

# -----------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------
def label_to_char(label):
    if 0 <= label <= 9:
        return str(label)
    elif 10 <= label <= 35:
        return chr(label - 10 + ord('A'))
    elif 36 <= label <= 61:
        return chr(label - 36 + ord('a'))
    return "?"

def get_char_type(label):
    if 0 <= label <= 9:
        return "Digit"
    elif 10 <= label <= 35:
        return "Uppercase Letter"
    elif 36 <= label <= 61:
        return "Lowercase Letter"
    return "Unknown"

def get_confidence_color(confidence):
    if confidence >= 0.8:
        return "🟢"
    elif confidence >= 0.5:
        return "🟡"
    return "🔴"


def preprocess_image(img_data, show_steps=False):
    steps = {}
    
    try:
        if isinstance(img_data, np.ndarray):
            img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
            img_gray = img.convert('L')
            steps['original'] = img_gray.copy()

            bbox = img_gray.getbbox()
            if bbox is None:
                st.warning("⚠️ Please draw a character first.")
                return None, None

            padding = 5
            bbox = (max(0, bbox[0]-padding), max(0, bbox[1]-padding), min(img_gray.width, bbox[2]+padding), min(img_gray.height, bbox[3]+padding))
            cropped = img_gray.crop(bbox)
            steps['cropped'] = cropped.copy()

            cropped = ImageEnhance.Contrast(cropped).enhance(1.5)
            steps['enhanced'] = cropped.copy()

            cropped.thumbnail((20,20), Image.Resampling.LANCZOS)
            steps['resized'] = cropped.copy()

            new_img = Image.new('L', (28, 28), 0)
            w, h = cropped.size
            new_img.paste(cropped, ((28-w)//2, (28-h)//2))
            img = new_img
        else:
            img = Image.open(img_data).convert('L')
            steps['original'] = img.copy()
            img = ImageEnhance.Contrast(img).enhance(1.3)
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            img = ImageOps.invert(img)
            steps['processed'] = img.copy()
        
        steps['final'] = img.copy()
        img_array = np.array(img) / 255.0
        return img_array.reshape(-1, 28, 28, 1), steps if show_steps else None
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None, None


def get_contextual_prediction(pred_array, context_type):
    top_indices = np.argsort(pred_array)[::-1]
    context_map = {"Digit (0-9)": "Digit", "Uppercase Letter (A-Z)": "Uppercase Letter", "Lowercase Letter (a-z)": "Lowercase Letter"}
    target_type = context_map.get(context_type, "Digit")
    
    for idx in top_indices:
        if get_char_type(idx) == target_type:
            return idx, pred_array[idx]
    return top_indices[0], pred_array[top_indices[0]]


# -----------------------------------------------------------
# UI
# -----------------------------------------------------------
st.title("🧠 EMNIST Digit & Alphabet Classifier")

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

if 'processed_steps' not in st.session_state:
    st.session_state.processed_steps = None

if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input")
    char_type_context = st.radio("🎯 Select Input Context:", ("Digit (0-9)", "Uppercase Letter (A-Z)", "Lowercase Letter (a-z)"))

    with st.expander("⚙️ Advanced Options"):
        show_preprocessing = st.checkbox("Show preprocessing steps", value=False)
        canvas_stroke_width = st.slider("Canvas stroke width", 8, 20, 12)

    tab1, tab2 = st.tabs(["✍️ Draw Character", "📤 Upload Image"])

    with tab1:
        canvas_result = st_canvas(
            stroke_width=canvas_stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )

        colA, colB = st.columns(2)
        if colA.button("🎯 Predict Drawing", use_container_width=True):
            processed_img, steps = preprocess_image(canvas_result.image_data, show_steps=show_preprocessing)
            if processed_img is not None:
                st.session_state.prediction = model.predict(processed_img)
                st.session_state.processed_steps = steps

        if colB.button("🧹 Clear", use_container_width=True):
            st.session_state.prediction = None
            st.session_state.canvas_key += 1
            st.rerun()

    with tab2:
        uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        if uploaded and st.button("🎯 Predict Upload", use_container_width=True):
            processed_img, steps = preprocess_image(uploaded, show_steps=show_preprocessing)
            if processed_img is not None:
                st.session_state.prediction = model.predict(processed_img)
                st.session_state.processed_steps = steps

with col2:
    st.subheader("🔍 Prediction Results")
    if st.session_state.prediction is None:
        st.info("👈 Predict to see results.")
    else:
        pred_array = st.session_state.prediction[0]
        idx, confidence = get_contextual_prediction(pred_array, char_type_context)
        char = label_to_char(idx)

        st.markdown(f"""
        <div style='text-align: center; font-size: 3rem;'> {char} </div>
        <p style='text-align:center;'>Confidence: {confidence*100:.2f}%</p>
        """, unsafe_allow_html=True)

        st.write("---")
        st.write("📊 Top Predictions:")
        top5 = np.argsort(pred_array)[-5:][::-1]

        for i in top5:
            st.write(f"{label_to_char(i)} → {pred_array[i]*100:.2f}%")

# -----------------------------------------------------------
# Footer
# -----------------------------------------------------------
st.write("---")
st.caption("Made with ❤️ using Streamlit | CNN EMNIST Model")
