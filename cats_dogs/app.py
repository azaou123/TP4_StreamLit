# animal_detector.py
import streamlit as st
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import re

# Title of the app
st.title("Enhanced Animal Detector")

st.write("Upload an image, and the app will identify if it contains an animal.")

# Uploading an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Prepare the image for the model
    img = img.resize((224, 224))  # Resize to match model's expected input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Load the pre-trained MobileNet model
    model = mobilenet_v2.MobileNetV2(weights="imagenet")

    # Perform prediction
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=5)[0]  # Checking top 5 predictions for better accuracy

    # Display top predictions
    st.write("Top predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i+1}. **{label}**: {score:.4f}")

    # Define a broader set of animal-related keywords
    animal_keywords = [
        "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "lion", "tiger",
        "monkey", "penguin", "dolphin", "wolf", "fox", "rabbit", "mouse", "rat", "deer", "goat", "camel", "leopard",
        "cheetah", "jaguar", "panther", "raccoon", "koala", "kangaroo", "panda", "crocodile", "alligator", "shark",
        "fish", "whale", "octopus", "eagle", "hawk", "owl", "duck", "goose", "chicken", "rooster", "pigeon", "parrot",
        "swan", "frog", "turtle", "lizard", "snake", "buffalo", "bison", "antelope", "bat", "beaver", "porcupine",
        "armadillo", "meerkat", "mole", "otter", "seal", "walrus", "narwhal", "manatee", "orca", "stingray", "jellyfish",
        "seahorse", "lobster", "crab", "shrimp", "clam", "oyster", "starfish", "urchin", "platypus", "echidna", "hedgehog",
        "flamingo", "peacock", "sparrow", "falcon", "woodpecker", "heron", "crow", "raven", "bluejay", "sparrowhawk",
        "quail", "canary", "macaw", "cockatoo", "parakeet", "budgie", "vulture", "mongoose", "weasel", "ermine", "lynx",
        "bobcat", "cougar", "hyena", "tapir", "sloth", "aardvark", "gazelle", "ibex", "muskox", "guinea pig", "hamster",
        "chinchilla", "beetle", "butterfly", "bee", "ant", "ladybug", "caterpillar", "grasshopper", "cricket", "firefly",
        "dragonfly", "scorpion", "tarantula", "tick", "flea", "fly", "mosquito", "wasp", "moth", "slug", "snail",
        "squid", "anchovy", "sardine", "trout", "salmon", "bass", "pike", "eel", "barracuda", "manta ray", "hammerhead",
        "mackerel", "swordfish", "clownfish", "sealion", "otter", "alpaca", "llama", "hedgehog", "hedgehog",
        "dingo", "peafowl", "fennec", "caracal", "ibis", "toucan", "cassowary", "turkey", "emu", "boar", "pig",
        "hedgehog", "jackal", "bandicoot", "quokka", "kudu", "okapi", "oribi", "vicuna", "opossum", "lynx", "margay",
        "ocelot", "serval", "pelican", "nightjar", "buzzard", "skunk", "squirrel", "chipmunk", "booby", "myna", "kiwi"
    ]


    # Set a minimum confidence threshold (e.g., 0.2)
    confidence_threshold = 0.2

    # Check if any of the top predictions match an animal keyword and get the animal name
    detected_animal = None
    for _, label, score in decoded_predictions:
        # Use regular expressions to match any keyword in the label
        if score >= confidence_threshold and any(re.search(keyword, label, re.IGNORECASE) for keyword in animal_keywords):
            detected_animal = label
            break

    # Display result based on whether an animal was detected
    if detected_animal:
        st.success(f"An animal has been detected in the image! It's a **{detected_animal}**.")
    else:
        st.warning("No animal detected in the image.")
