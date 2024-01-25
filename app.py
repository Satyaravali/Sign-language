import streamlit as st
from PIL import Image
import numpy as np
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import os.path

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Load pre-trained model
from tensorflow.keras.models import load_model

model = load_model('SignL.h5')
def predict(image):    
    img_array = image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)

    # Manually map the class index to your custom class labels
    custom_class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z","del","nothing","space"]  # Replace with your actual class labels
    predicted_class_label = custom_class_labels[predicted_class_index]


    return predicted_class_label
def main():
    st.title("Image Classification with MobileNet ")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make predictions on the uploaded image
        predictions = predict(image)

        st.subheader("Predictions:")
        st.write(predicted_class_label)


if _name_ == "_main_":
    main()
