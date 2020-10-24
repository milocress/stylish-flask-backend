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
from werkzeug.utils import secure_filename

import style_video

dirname = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(dirname, 'static/')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}
FILE_TYPE = {"jpg": "image", "jpeg": "image", "png": "image", "mp4": "video"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def show_index():
  return render_template('index.html')

@app.route('/url_form')
def get_url_form():
    return render_template('url_form.html')

@app.route('/upload_form')
def get_upload_form():
    return render_template('upload_form.html')

@app.route('/url_form', methods=['POST'])
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

#================================
#  merged from dwu upload.py

# Copied from flask documentation
def allowed_file(filename):
    return '.' in filename and \
           extension(filename) in ALLOWED_EXTENSIONS

def extension(filename):
	return filename.rsplit('.', 1)[1].lower()

@app.route('/upload_form', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(filepath)
            if FILE_TYPE[extension(file.filename)] == 'video':
                return render_template("video.html", filename = filename, filetype = FILE_TYPE[extension(filename)])
            return render_template("image.html", filename = filename, filetype = FILE_TYPE[extension(filename)])

    return render_template("upload_form.html")

app.run()
