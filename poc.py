from flask import Flask, send_file, request, render_template
import json
import requests
from PIL import Image
import tensorflow as tf 
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_hub as hub
import io
import functools

app = Flask(__name__)

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image


@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
  if img.max() > 1.0:
    img = img / 255.
  if len(img.shape) == 3:
    img = tf.stack([img, img, img], axis=-1)
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

@app.route('/form')
def my_form():
    return render_template('url_form.html')

@app.route('/form', methods=['POST'])
def my_form_post():
    content = request.form['content url']
    style = request.form['style url']

    api_url = 'http://localhost:5000/image_urls'
    data = {'content': content,'style':style}
    r = requests.get(url=api_url, json=data)
    file_object = io.BytesIO(r._content)

    return send_file(file_object, mimetype='image/PNG')

@app.route('/image_urls')
def image_urls():
    content_image_url = request.json['content'] #if key doesn't exist, returns None
    style_image_url = request.json['style']
    output_image_size = 384  # @param {type:"integer"}

    # The content image size can be arbitrary.
    content_img_size = (output_image_size, output_image_size)
    # The style prediction model was trained with image size 256 and it's the
    # recommended image size for the style image (though, other sizes work as
    # well but will lead to different results).
    style_img_size = (256, 256)  # Recommended to keep it at 256.

    content_image = load_image(content_image_url, content_img_size)
    style_image = load_image(style_image_url, style_img_size)
    style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')
    # show_n([content_image, style_image], ['Content image', 'Style image'])

    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)

    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]


    img = tf.keras.preprocessing.image.array_to_img(
        tf.squeeze(stylized_image).numpy(), data_format=None, scale=True, dtype=None
    )
    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    img.save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start    
    file_object.seek(0)

    return send_file(file_object, mimetype='image/PNG')


app.run()