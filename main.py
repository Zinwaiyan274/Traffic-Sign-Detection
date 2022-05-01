import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

st.title('Traffic Sign Detection')
st.markdown('''
Try Anything
''')

@st.cache(allow_output_mutation=True)

def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction_percentage = model.predict(data)
    prediction=prediction_percentage.round()
    
    return  prediction,prediction_percentage


uploaded_file = st.file_uploader("Choose a Image...")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded file', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label,perc = teachable_machine_classification(image, 'Final.h5')
    st.write(label)
    a,b =np.array(label)
    st.write(a,b)
    if label.any() == 0:
     st.write("Giveway",perc)
    if label.any() == 1:
      st.write("NoEntry",perc)   
    if label.any() == 2:
     st.write("NoHorn",perc)   
    if label.any() == 3:
     st.write("Roundabout",perc)  
    if label.any() == 4:
     st.write("Stop",perc)
    else:
     st.write("Have a great day")
   

#if label == 0:
#    st.write("Giveway",perc)
#if label == 1:
#    st.write("NoEntry",perc)   
#if label == 2:
#    st.write("NoHorn",perc)   
#if label == 3:
#    st.write("Roundabout",perc)  
#if label == 4:
#    st.write("Stop",perc)
#else:
#    st.write("Have a great day")
