import io
import os
import shutil
import re
import cv2
import base64
from ultralytics import YOLO
from supervision.tools.detections import Detections
import pytesseract
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_image_select import image_select

MODEL = ""


def image_ai(image, source):
    global MODEL
    numpy_array = np.array(image)
    opencv_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)

    if source == "icons/1.png":  # VK
        MODEL = "models/best_vk_small.pt"

    if source == "icons/2.png":  # Дзен
        MODEL = "models/best_zn_small.pt"

    if source == "icons/3.png":  # YouTube
        MODEL = "models/best_yt_small.pt"

    if source == "icons/4.png":  # RuTube
        pass

    if source == "icons/5.png":  # Yappy
        pass

    if source == "icons/6.png":  # Telegram
        MODEL = "models/best_tg_small.pt"

    if source == "icons/7.png":  # OK
        pass

    if os.path.exists('cash'):
        shutil.rmtree('cash')

    if not os.path.exists('cash'):
        os.makedirs('cash')
        os.makedirs('cash/imgs')

    model = YOLO(MODEL)
    model.fuse()

    CLASS_NAMES_DICT = model.names
    # Проходим по всем файлам в папке
    try:
        results = model(opencv_image, imgsz=1280)
        detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )

        # format custom labels and frames
        label_frame_list = []
        for detection in detections:
            x1, y1, x2, y2 = detection[0]  # Get bounding box coordinates
            cropped_frame = opencv_image[int(y1):int(y2), int(x1):int(x2)]  # Crop frame
            label = CLASS_NAMES_DICT[detection[2]]  # Get label
            label_frame_list.append((label, cropped_frame))

        image_np_array = label_frame_list[0][1]
        image_pil = Image.fromarray(image_np_array)
        kpi = pytesseract.image_to_string(image_pil, lang='rus')
    except:
        kpi = 'Invalid'
    return kpi


def download_link(object_to_download, download_filename, link_text):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{link_text}</a>'
    return href


def load_images_from_zip(zip_file):
    images = []
    with zipfile.ZipFile(zip_file) as archive:
        for file_info in archive.infolist():
            if file_info.filename.endswith('/'):
                continue
            if file_info.filename.split('/')[-1].split('.')[-1].lower() not in correct_types:
                invalid_img.append(file_info.filename)

            with archive.open(file_info) as file:
                try:
                    image = Image.open(file)
                    image.load()
                    images.append((image, file_info.filename))
                except IOError:
                    pass
    return images


def load_images():
    uploaded_files = st.file_uploader(label="Выберите изображения или архив с изображениями для распознавания",
                                      accept_multiple_files=True, type=["jpg", "png", 'bmp', 'jpeg', 'zip', 'heic'])
    images = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".zip"):
                zip_images = load_images_from_zip(uploaded_file)
                # for image, filename in zip_images:
                #    st.image(image, caption=filename)
                images.extend(zip_images)
            else:
                image_data = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(image_data))
                # st.image(image, caption=uploaded_file.name)
                images.append((image, uploaded_file.name))
    if len(images) == 0:
        st.warning("No images were uploaded.")
        return None
    return images



invalid_img = []
correct_types = ["jpg", "png", 'bmp', 'jpeg', 'heic']

button_clicked = False
col1, col2, col3, col4, col5 = st.columns(5)
if not button_clicked:
    if col3.button("Отправить") and MODEL != "": button_clicked = True

if not button_clicked:
    st.markdown("<h1 style='text-align: center;'>Загрузка данных</h1>", unsafe_allow_html=True)

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
        ], use_container_width=False)

    if imgs == "icons/4.png" or imgs == "icons/5.png" or imgs == "icons/7.png":
        st.warning('В разработке')

    images = load_images()
    if images is not None:
        column_names = ['KPI', 'File name']
        df = pd.DataFrame(columns=column_names)
        st.write(f"Загружено {len(images)} изображений.")

        for img in invalid_img:
            new_row = ['Invalid', img]
            df.loc[len(df)] = new_row

        for img, name in images:
            if name.split('/')[-1].split('.')[-1].lower() in correct_types:
                result = image_ai(img, imgs)

            else:
                result = 'Invalid'
            new_row = [result, name]
            df.loc[len(df)] = new_row

        pattern = r'(\d+\.\d+|\d+)'

        df['kpi'] = df['kpi'].apply(lambda x: ' '.join(re.findall(pattern, x)) if (
                'invalid' not in x.lower() and re.search(pattern, x)) else 'invalid')

        if not os.path.exists('cash'):
            os.makedirs('cash')
            os.makedirs('cash/imgs')

        df.to_csv('cash/data.csv', index=False)

        top5 = df.loc[df['KPI'] != 'Invalid'].sort_values(by='KPI', ascending=False).head(5)['File name'].tolist()
        print(top5)
        with open('cash/top5.txt', 'w') as file:
            for item in top5:
                file.write(item + '\n')
        for one in top5:
            for img in images:
                if one == img[1]:
                    img[0].save(os.path.join('cash/imgs', one))

        print('Finished')

if button_clicked:
    st.markdown("<h1 style='text-align: center;'>Подготовка отчета</h1>", unsafe_allow_html=True)
    df = pd.read_csv('cash/data.csv')
    fdf = df.loc[df['KPI'] != 'Invalid']

    st.dataframe(df, hide_index=True, width=699)

    with st.expander("Доп. информация", False):
        max_value = fdf['KPI'].max()
        st.write(f"Максимальный KPI: {max_value}")
        min_value = fdf['KPI'].min()
        st.write(f"Минимальный KPI: {min_value}")
        st.write(f"Всего изображений {len(df)}")
        st.write(f"Не обработалось: {df['KPI'].value_counts().get('Invalid', 0)}")

    fig, ax = plt.subplots()
    ax.hist(fdf["KPI"], bins=15)
    ax.set_xlabel('KPI')
    ax.set_ylabel('Частота')
    ax.set_title('Гистограмма распределения KPI')
    st.pyplot(fig)

    with open('cash/top5.txt', 'r') as file:
        array = [line.strip() for line in file]
    st.markdown("<h2 style='text-align: center;'>Топ 5</h2>", unsafe_allow_html=True)

    for item in array:
        print(item)
        image = Image.open(f'cash/imgs/{item}')
        st.image(image, caption=item)

    st.markdown(download_link(df, "example.csv",
                              "<div style='text-align: center; color: grey; font-size: 34px;'>Скачать</div>"),
                unsafe_allow_html=True)