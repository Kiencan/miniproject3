import streamlit as st
import pyautogui
from PIL import Image
import numpy as np
import cv2
from predict import *


st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")
with st.container():
    st.title("Simple Optical Character Recognition")

    st.write("---")

    st.subheader("Upload your image here:")
    image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    show_file = st.empty()

    if not image:
        show_file.info("Please upload an image file: {}".format(" ".join(["jpg", "jpeg", "png"])))

    if image is not None:
        image_1 = Image.open(image)
        img_array_1 = np.array(image_1)
        st.image(image_1)



    st.write("---")
    col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9, col_10, col_11, col_12, = st.columns(12)

    
    if col_1.button("Reset", type="primary", use_container_width=True):
        pyautogui.hotkey("ctrl", "f5")

    

    if col_2.button("Predict by Logistic", type="secondary", use_container_width=True):
        st.write("Your results: ")
        new_img = process_logistic(img_array_1)
        st.image(new_img)


    if col_3.button("Predict by CNN", type="secondary", use_container_width=True):
        st.write("Your results: ")
        new_img1 = process_cnn(img_array_1)
        st.image(new_img1)




    