import os
import numpy as np
#import cv2
from tensorflow.keras.models import load_model
import streamlit as st

#MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
#if not os.path.isdir(MODEL_DIR):
#    os.system('runipy Final.ipynb')

#model = load_model('model')
# st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

st.title('Traffic Sign Detection')
st.markdown('''
Try Anything
''')

# data = np.random.rand(28,28)
# img = cv2.resize(data, (256, 256), interpolation=cv2.INTER_NEAREST)



uploaded_file = st.file_uploader("Choose a file")

if st.button('Try'):
    st.write("Done")

#if uploaded_file is not None:
#     
#    img = cv2.resize(uploaded_file, (150, 150))
#    rescaled = cv2.resize(img, (150, 150), interpolation=cv2.INTER_NEAREST)
#    st.write('Model Input')
#    st.image(rescaled)

#if st.button('Predict'):
#    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    val = model.predict(test_x.reshape(1, 150, 150))
#    st.write(f'result: {np.argmax(val[0])}')
#    st.bar_chart(val[0])
