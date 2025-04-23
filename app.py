import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
import os
from io import BytesIO
from pathlib import Path

# Download model from Google Drive (only once)
MODEL_URL = "https://drive.google.com/uc?id=10ZzSQ_Pn_2Xqai8ZYs3p_-_YRqmO6JlB"
MODEL_PATH = "fine_tuned_model.h5"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model. Please wait...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    st.success("Model downloaded!")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# UI
st.title("E-commerce Product Image Quality Predictor")
uploaded_file = st.file_uploader("Upload a Product Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array)[0][0]
    st.write(f"Prediction Score: {prediction:.2f}")

    if prediction >= 0.5:
        st.success("✅ Good Quality Image")
    else:
        st.warning("⚠ Poor Quality Image")