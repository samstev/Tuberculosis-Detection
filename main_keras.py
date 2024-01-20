import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# Load the model
model = load_model('./my_model.keras')  # Adjust the path

def predict_image(img, model):
    # Preprocess the image to fit your model's input size (150x150 in your case)
    img = img.resize((150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.

    # Make prediction
    prediction = model.predict(img)
    return prediction

# Streamlit app
st.title('TB Detection App')
st.write('Upload an X-ray image for TB detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Predict
    img = image.load_img(uploaded_file)
    prediction = predict_image(img, model)

    # Display prediction
    if prediction[0][0] > 0.5:
        st.write('Prediction: Normal')
    else:
        st.write('Prediction: Tuberculosis')
