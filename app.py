import streamlit as st
from PIL import Image, ImageOps
import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np


class_names = ["Healthy", "Maize Streak Virus"]

def load_best_model():
    return load_model('artifacts/model.h5')

def import_and_predict(image_data, model):
    size = (224,224)
    image = ImageOps.fit(image_data, size)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

st.markdown("## Maize Streak Disease Classification", unsafe_allow_html=True)

with st.spinner('Model is being loaded..'):
    best_model = load_best_model()

st.write("""
Despite the fact that the agricultural sector is a national economic development priority in sub-Saharan Africa, crop pests and diseases have been the challenge affecting major food security crops like maize. 
Maize Streak Disease which is caused by the Maize Streak Virus is regarded as the third most serious disease affecting maize in sub-Saharan Africa. 
""")

file = st.file_uploader("Please upload the image file", type=["jpg", "png"])

if file is None:
    st.text("File has not been uploaded yet.")
else: 
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, best_model)
    score = tf.nn.softmax(predictions[0])
    result_text = "This image most likely belongs to the <b>{}</b> class.".format(class_names[np.argmax(score)])
    st.markdown(result_text, unsafe_allow_html=True)