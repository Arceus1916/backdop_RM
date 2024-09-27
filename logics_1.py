import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os

# Load the pre-trained model (cv_model_v3.h5)
def load_trained_model():
    model_path = 'cv_model_v2.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file '{model_path}' was not found.")
    model = load_model(model_path)
    return model

import numpy as np

def classify_image(model, image_path):
    # Load and preprocess the image (128x128x3 expected)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to match the model's input size
    img = img.astype('float32') / 255.0  # Normalize image (optional)
    
    # Expand dimensions to simulate a batch size of 1
    img = np.expand_dims(img, axis=0)  # Shape becomes (1, 128, 128, 3)

    # Make the prediction
    prediction = model.predict(img)
    
    # Get the predicted class index
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Convert predicted class index to tag
    if predicted_class == 0:
        return 'violence'
    elif predicted_class == 1:
        return 'toilets'
    elif predicted_class == 2:
        return 'coaches'
    else:
        return 'unknown'

# Simulated data (Route1, Route2, Route3)
Route1 = {
    '8116841775':['TT025', 'CRPF456', 'RI087'],
    '8114179135': ['TT628', 'CRPF164', 'RI069'],
    '8118694523': ['TT924', 'CRPF548', 'RI007']
}

Route2 = {
    '4205524732': ['TT001', 'CRPF784', 'RI019'],
    '4202926182': ['TT002', 'CRPF451', 'RI049'],
    '4203648179': ['TT069', 'CRPF696', 'RI027']
}

Route3 = {
    '4402924117': ['TT003', 'CRPF856', 'RI096'],
    '4402947713': ['TT009', 'CRPF741', 'RI023'],
    '4402628697': ['TT015', 'CRPF720', 'RI048']
}
combined_routes = {**Route1, **Route2, **Route3}


def get_responsible_person(tag, pnr_number):
    # Check if the PNR exists in the dataset
    if pnr_number not in combined_routes:
        return "PNR number not found"

    # Retrieve the values for the given PNR number
    values = combined_routes[pnr_number]

    # Handle based on tag
    if tag == 'violence':
        return [values[1], values[2]]  # CRPFXXX and RIXXX
    elif tag == 'coaches' or tag == 'washroom':
        return [values[0], values[2]]  # TTXXX and RIXXX
    else:
        return "Invalid tag"
