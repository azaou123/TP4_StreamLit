# cat_dog_app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Title of the app
st.title("Cat and Dog Classifier")

st.write("Upload an image, and the app will classify it as a cat or dog.")

# Load the trained model
model = load_model('saved_model/cat_dog_model.h5')

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize to match model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# Upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Predict the class
    prediction = model.predict(img_array)
    class_label = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction[0][0] if prediction > 0.5 else 1 - prediction[0][0]

    # Display the prediction
    st.write(f"**Prediction**: This is a **{class_label}** with **{confidence:.2%}** confidence.")
