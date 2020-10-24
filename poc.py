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

import style_video

app = Flask(__name__)

@app.route('/')
def show_index():
  return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def show_form():
  return render_template("url_form.html")

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

    content_image_path = style_video.get_image_path_from_url(content_image_url)
    style_image_path = style_video.get_image_path_from_url(style_image_url)

    content_image = style_video.get_content_image_from_path(content_image_path)
    style_image = style_video.preprocesses_style_image(style_image_path)

    img = style_video.get_style_transfer(content_image, 0, style_image, True)

    file_object = io.BytesIO()
    img.save(file_object, 'PNG')
    file_object.seek(0)
    
    return send_file(file_object, mimetype='image/PNG')

app.run()
