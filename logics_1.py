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
    '1001': ['tt_01', 'crpf_01', 'route_incharge_01'],
    '1002': ['tt_02', 'crpf_02', 'route_incharge_01'],
    '1003': ['tt_03', 'crpf_03', 'route_incharge_01']
}

Route2 = {
    '2001': ['tt_04', 'crpf_04', 'route_incharge_02'],
    '2002': ['tt_05', 'crpf_05', 'route_incharge_02'],
    '2003': ['tt_06', 'crpf_06', 'route_incharge_02']
}

Route3 = {
    '3001': ['tt_07', 'crpf_07', 'route_incharge_03'],
    '3002': ['tt_08', 'crpf_08', 'route_incharge_03'],
    '3003': ['tt_09', 'crpf_09', 'route_incharge_03']
}

def get_responsible_person(tag, pnr_number):
    if tag == 'violence':
        crpf_id = "crpf_" + pnr_number[-2:]  # example logic for CRPF ID
        route_incharge = "route_incharge_" + pnr_number[-2:]  # example logic for route incharge
        return {'responsible_person': crpf_id, 'route_incharge': route_incharge}
    else:
        tt_id = "tt_" + pnr_number[-2:]  # example logic for TT ID
        route_incharge = "route_incharge_" + pnr_number[-2:]  # example logic for route incharge
        return {'responsible_person': tt_id, 'route_incharge': route_incharge}
