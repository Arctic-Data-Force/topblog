import io
import streamlit as st
from PIL import Image
from streamlit_image_select import image_select
import zipfile




def load_images_from_zip(zip_file):
    images = []
    with zipfile.ZipFile(zip_file) as archive:
        for file_info in archive.infolist():
            if file_info.filename.endswith('/'):
                continue  # skip directories
            with archive.open(file_info) as file:
                try:
                    image = Image.open(file)
                    image.load()  # force file to be read
                    images.append((image, file_info.filename))
                except IOError:
                    # Not an image file, skip.
                    pass
    return images

def load_images():
    uploaded_files = st.file_uploader(label="Выберите изображение или архив с изображениями для распознавания", accept_multiple_files=True)
    
    images = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".zip"):
                # Если загруженный файл - архив, извлекаем изображения
                zip_images = load_images_from_zip(uploaded_file)
                for image, filename in zip_images:
                    st.image(image, caption=filename)
                images.extend(zip_images)
            else:
                # Если загружен файл с изображением
                image_data = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(image_data))
                st.image(image, caption=uploaded_file.name)
                images.append((image, uploaded_file.name))
    if len(images) == 0:
        st.warning("No images were uploaded.")
        return None
    return images

st.title("Тестовое приложение")

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

# Пример использования:
images = load_images()
if images is not None:
    st.write(f"Загружено {len(images)} изображений.")



result = st.button("Обработать изображение",)
