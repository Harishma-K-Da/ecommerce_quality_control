import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model
model = tf.keras.models.load_model("ecommerce_model.keras")  # Use your actual model filename

# UI setup
st.set_page_config(page_title="Product Quality Checker", layout="centered")
st.title("🛍 E-commerce Product Image Quality Checker")
st.write("Upload a product image (shirt, gown, jeans, shoes, etc.) to evaluate its quality.")

# Upload file
uploaded_file = st.file_uploader("📤 Upload a product image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    # Preprocessing
    image = image.resize((150, 150))  # adapt size as per your model
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    confidence = prediction * 100

    # Emoji-based quality result
    if confidence >= 80:
        st.success(f"🌟 Looks Beautiful** – Confidence: {confidence:.2f}%")
    elif confidence >= 75:
        st.info(f"✅ Good Quality** – Confidence: {confidence:.2f}%")
    elif confidence < 30:
        st.error(f"❌ Damaged** – Confidence: {100 - confidence:.2f}%")
    else:
        st.warning(f"⚠ Average Quality** – Confidence: {confidence:.2f}%")