# Image Super Resolution (Daisi Hackathon)

Python function as a web service to enhance low resolution image to high quality image.

The service is based on **Single Image Super Resolution (SISR)** deep learning model which is available on Open Model Zoo, check this [link](https://docs.openvino.ai/latest/omz_models_model_single_image_super_resolution_1032.html) for more info.

**Super Resolution** is the process of enhancing the quality of an image by increasing the pixel count using deep learning.

* The model (Neural Network) expects inputs with a width of **480**, height of **270**.
* The model returns images with a width of **1920**, height of **1080**.
* The image sides are upsampled by a factor **4**. The new image is **16** times as large as the original image.

In brief:
* Input image should be: **480x270** resolution.
* Output image : **1920x1080** resolution.

