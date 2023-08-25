import io
import streamlit as st
from PIL import Image
import numpy as np

def load_image():
    uploaded_file = st.file_uploader(label="Выберите изображение для распознавания")
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None
    
st.title("Тестовое приложение")

from streamlit_image_select import image_select

imgs = image_select(
    label="Выберите платформу",
    images=[
        "icons/1.png",
        "icons/2.png",
        "icons/3.png",
        "icons/4.png",
        "icons/5.png",
        "icons/6.png",
        "icons/7.png",
    ],
    
    use_container_width=False,
)

img = load_image()
result = st.button("Обработать изображение")
