#Import the required Libraries
import streamlit as st
from keras.models import load_model
from tensorflow.keras.preprocessing import image as image_utils
import numpy as np
from io import BytesIO

# Add a title and intro text
st.markdown("<h1 style='text-align: center; color: white;'>🌿كزبرة ولا بقدونس؟</h1>", unsafe_allow_html=True)
input_image = st.file_uploader('Upload an image of a parsley or coriander')

@st.experimental_singleton
def load_model_from_path(suppress_st_warning=True):
    return load_model('Kozbra_Ba2dons_accurate.h5')
test  = load_model_from_path()

if input_image:
    img = input_image.read()
    image = image_utils.load_img(BytesIO(img), color_mode="rgb", target_size=(256,256))
    st.image(image)
    image = image_utils.img_to_array(image)
    
    image = image.reshape(1, 256,256,3)

    prediction = test.predict(image)
    if prediction > 0.03:
       final = "بقدونس"
    elif prediction <0.03:
       final = "كزبرة"    

    
    st.markdown(f"<h1 style='text-align: center; color: green; font-size:40px'>{final}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: white; font-size:40px'>{prediction[0]}</h1>", unsafe_allow_html=True)
