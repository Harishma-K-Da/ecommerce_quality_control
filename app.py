import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Load the fine-tuned model
model = tf.keras.models.load_model(r"C:\Users\DELL\Downloads\ecommerce_quality_control\data\fine_tuned_model.h5")
IMG_SIZE = (224, 224)
class_names = ['Blurry', 'Correct']  # Modify if you have different classes

def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    return predicted_class, confidence

# Streamlit UI
st.title("E-commerce Product Image Quality Checker")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    predicted_class, confidence = predict_image(image)

    st.markdown(f"### Prediction: {predicted_class}")
    st.markdown(f"### Confidence Score: {confidence:.2f}%")