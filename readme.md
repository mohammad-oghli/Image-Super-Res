# Image Super Resolution (Daisi Hackathon)

![Super_Figure](../Superres_Figure.png.png)

Python function as a web service to enhance low resolution image to high quality image.

The service is based on **Single Image Super Resolution (SISR)** deep learning model which is available on Open Model Zoo, check this [link](https://docs.openvino.ai/latest/omz_models_model_single_image_super_resolution_1032.html) for more info.

**Super Resolution** is the process of enhancing the quality of an image by increasing the pixel count using deep learning.

* The model (Neural Network) expects inputs with a width of **480**, height of **270**.
* The model returns images with a width of **1920**, height of **1080**.
* The image sides are upsampled by a factor **4**. The new image is **16** times as large as the original image.

In brief:
* Input image should be: **480x270** resolution.
* Output image : **1920x1080** resolution.

You can use image samples in the **/images** directory to test it on the model.

### How to call it

-Load the Daisi
<pre>
import matplotlib.pyplot as plt
import pydaisi as pyd
image_super_resolution = pyd.Daisi("oghli/Image Super Resolution")
</pre>

-Call the `image_super_resolution` end point, passing the image source to enhance it, you can pass image source either from **images/** directory or from valid **url** of the image
<pre>
#image_source = "https://i.imgur.com/R5ovXDO.jpg"
image_source = "images/witcher.jpg"
result = image_super_resolution.cv_superresolution(image_source).value
result
</pre>
-It will return two **np arrays** representing:
* Original image reshaped to the target resolution of the model
* Super resolution image

you can save result images in variables
<pre>
origin_image = result[0] 
superresolution_image = result[1] 
</pre>

Then display super resolution image
<pre>
plt.imshow(superresolution_image)
</pre>

Also, you can display comparison figure for image before and after enhancement by excuting the following code
<pre>
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))
ax[0].imshow(origin_image)
ax[1].imshow(superresolution_image)
ax[0].set_title("Origin")
ax[1].set_title("Superresolution")
</pre>

Function `st_ui` included in the app to render the user interface of the application endpoints.

Check the research paper reference for more info about super resolution model:

https://arxiv.org/abs/1807.06779