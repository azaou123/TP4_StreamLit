import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Charger le modèle
model = tf.keras.models.load_model("cifar10_model.h5")

# Labels de classes CIFAR-10
class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

st.title("CIFAR-10 Image Classifier")

st.write("Téléchargez une image de 32x32 pixels, et l’application prédira sa classe.")

uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Prétraiter l'image
    img = Image.open(uploaded_file).resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prédire la classe de l'image
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.image(img, caption="Image téléchargée", use_column_width=True)
    st.write(f"Classe prédite : **{predicted_class}**")
