import streamlit as st
import os
import time
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from openvino.inference_engine import IECore
from helper import load_image, write_text_on_image, convert_result_to_image, to_rgb, pil_to_bytes
from model_Inference import do_inference

# Global model configuration
# Setting
DEVICE = "CPU"
# 1032: 4x superresolution, 1033: 3x superresolution
MODEL_FILE = "model/single-image-super-resolution-1032.xml"
model_name = os.path.basename(MODEL_FILE)
model_xml_path = Path(MODEL_FILE).with_suffix(".xml")

# Load the Superresolution Model
ie = IECore()
net = ie.read_network(model=str(model_xml_path))
exec_net = ie.load_network(network=net, device_name=DEVICE)


def cv_superresolution(image_source):
    '''
    Enhance low resolution image using deep learning (SISR) model
    :param
    image_source(str): Valid image url or image object of the input image

    :return
    bicubic_image(np.ndarray): Reshaped input image to the model target resolution
    - superresolution_image(np.ndarray): Super resolution image of the input image
    '''
    OUTPUT_PATH = Path("output/")
    os.makedirs(str(OUTPUT_PATH), exist_ok=True)
    full_image = load_image(image_source)
    # Uncomment these lines to load a raw image as BGR
    # import rawpy
    # with rawpy.imread(IMAGE_PATH) as raw:
    #     full_image = raw.postprocess()[:,:,(2,1,0)]

    # plt.imshow(to_rgb(full_image))
    # print(f"Showing full image with width {full_image.shape[1]} " f"and height {full_image.shape[0]}")
    # plt.show()
    if full_image is not None:
        return do_inference(full_image, exec_net)

    return "Sorry, loading image failed"


def st_ui():
    '''
    Render the User Interface of the application endpoints
    '''
    st.title("Image Super Resolution")
    st.caption("Image Quality Enhancement")
    st.info("Single Image Super Resolution (SISR) Implementation by Oghli")
    container_hints = st.empty()
    with container_hints.container():
        st.markdown("* The model expects inputs with a width of 480, height of 270")
        st.markdown("* The model returns images with a width of 1920, height of 1080")
        st.markdown("* It enhances the resolution of the input image by a factor of 4")

    st.sidebar.subheader("Upload image to enhance its quality")
    uploaded_image = st.sidebar.file_uploader("Upload image", type=["png", "jpg"],
                                          accept_multiple_files=False, key=None, help="Image to enhance its quality")
    gen_btn = st.sidebar.button("Generate")
    col1, col2 = st.columns(2)
    s_msg = st.empty()
    example_image = load_image('images/space.jpg')
    ex_sub = col1.subheader("Image Sample")
    example = col1.image(to_rgb(example_image), use_column_width=True)
    if uploaded_image:
        s_msg = st.sidebar.success("Image uploaded successfully")
        example.empty()
        ex_sub.empty()

    if gen_btn:
        s_msg.empty()
        example.empty()
        ex_sub.empty()
        full_image = 'images/space.jpg'
        if uploaded_image:
            full_image = uploaded_image
        container_hints.empty()
        # with st.spinner('Generating Super Resolution Image ...'):
        full_bicubic_image, full_superresolution_image = cv_superresolution(full_image)
        col1.subheader("Original Image")
        col1.image(full_bicubic_image, use_column_width=True)
        col2.subheader("SISR Image")
        col2.image(full_superresolution_image, use_column_width=True)
        byte_super_img = pil_to_bytes(full_superresolution_image)
        st.download_button(label="Download Result", data=byte_super_img,
                           file_name="superres_image_4x.jpeg", mime="image/jpeg")
        st.header("Animated GIF Comparison")
        image_super = write_text_on_image(image=full_superresolution_image, text="SUPER")
        image_bicubic = write_text_on_image(image=full_bicubic_image, text="ORIGIN")

        img_array = [image_bicubic, image_super]
        image_gif = st.empty()
        i = 0
        while True:
            image_gif.image(img_array[i])
            i = not i
            time.sleep(2)


if __name__ == "__main__":
    # render the app using streamlit ui function
    st_ui()
    # image_source = "images/mass_effect.jpg"
    # #image_source = "https://i.imgur.com/R5ovXDO.jpg"
    # full_bicubic_image, full_superresolution_image = cv_superresolution(image_source)
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))
    # ax[0].imshow(full_bicubic_image)
    # ax[1].imshow(full_superresolution_image)
    # ax[0].set_title("Origin")
    # ax[1].set_title("Superresolution")
    # plt.show()
