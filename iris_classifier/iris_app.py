# iris_app.py
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("iris_model.pkl")

# Title and instructions
st.title("Iris Flower Species Classifier")
st.write("Enter the features of the Iris flower to predict its species.")

# Define feature input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Predict button
if st.button("Predict Species"):
    # Create an array from user inputs
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)

    # Map the prediction to species name
    species = ["Setosa", "Versicolor", "Virginica"]
    predicted_species = species[prediction[0]]
    confidence = prediction_proba[0][prediction[0]] * 100

    # Display the result
    st.write(f"Predicted Species: **{predicted_species}**")
    st.write(f"Prediction Confidence: **{confidence:.2f}%**")
