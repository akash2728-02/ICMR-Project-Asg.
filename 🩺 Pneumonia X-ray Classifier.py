# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 21:25:16 2025

@author: akash
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pickle
from PIL import Image
import io

# Load the model
with open('C:/Users/akash/Job Assignment/ICMR Chandigrah/inception_model.pkl', "rb") as f:
    model = pickle.load(f)

# UI
st.title("🩺 Pneumonia X-ray Classifier")
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.stack([img_array]*3, axis=-1)
    img_array = tf.image.resize(img_array, [224, 224]).numpy() 
    img_array=img_array/ 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Predict
    prediction = model.predict(img_array)
    label = "🫁 Pneumonia" if prediction > 0.5 else "✅ Normal"
    st.markdown(f"### Prediction: {label}")
