import os
#import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'Final.h5')
if not os.path.isdir(MODEL_DIR):
    os.system('runipy Final.ipynb')

model = load_model('Final.h5')

st.title('Traffic Sign Detection')
st.markdown('''
Try Anything
''')




uploaded_file = st.file_uploader("Choose a file")

#if st.button('Try'):
#    st.write("Done")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    img = cv2.imread(bytes_data)
    #rescaled = cv2.resize(img, (150, 150), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(img)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = model.predict(test_x)
    st.write(f'result: {np.argmax(val[0])}')
    st.bar_chart(val[0])
    #dataframe = pd.read_csv(uploaded_file)
    #st.write(dataframe)
