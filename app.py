import io
import json
import os

import click
import numpy as np
import requests
from flask import Flask, render_template, request, send_file
from PIL import Image
from werkzeug.utils import secure_filename

from fast_neural_style_pytorch.stylize import stylize
from style_video import fast_style_transfer_video_file

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
dirname = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(dirname, "static/")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4"}
FILE_TYPE = {"jpg": "image", "jpeg": "image", "png": "image", "mp4": "video"}
REMOTE_URL = "https://stylish-videos.herokuapp.com"
LOCAL_URL = "http://localhost:5000"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


@app.route("/")
def show_index():
    return render_template("index.html")


@app.route("/upload_form")
def get_upload_form():
    return render_template("upload_form.html")


# Copied from flask documentation
def allowed_file(filename):
    return "." in filename and extension(filename) in ALLOWED_EXTENSIONS


def extension(filename):
    return filename.rsplit(".", 1)[1].lower()


@app.route("/fast_form", methods=["GET", "POST"])
def fast_upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file1" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file1 = request.files["file1"]
        stylepath = request.form["style"]
        # if user does not select file, browser also
        # submit an empty part without filename
        if file1.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file1 and allowed_file(file1.filename):
            filename1 = secure_filename(file1.filename)
            filepath1 = os.path.join(app.config["UPLOAD_FOLDER"], filename1)
            file1.save(filepath1)
            print(filepath1)
            data = {"content": filepath1, "style": stylepath, "lite": app.use_tflite}
            content_filetype = FILE_TYPE[extension(file1.filename)]

            if content_filetype == "video":
                r = requests.get(url=f"{app.api_url}/fast_video_uploads", json=data)
                return render_template(
                    "video.html",
                    filename=r._content.decode("ascii"),
                    filetype=content_filetype,
                )
            else:
                r = requests.get(url=f"{app.api_url}/fast_image_uploads", json=data)
                file_object = io.BytesIO(r._content)

                return send_file(file_object, mimetype="image/PNG")

    return render_template("fast_form.html")


@app.route("/fast_image_uploads")
def fast_image_upload():
    content_path = request.json["content"]
    print(content_path, flush=True)
    if content_path == None:
        content_path = "fast_neural_style_pytorch/images/tokyo2.jpg"
    style_path = request.json["style"]
    img = tf.keras.preprocessing.image.array_to_img(
        stylize(content_path, style_path), data_format=None, scale=True, dtype=None
    )
    file_object = io.BytesIO()
    img.save(file_object, "PNG")
    file_object.seek(0)
    return send_file(file_object, mimetype="image/PNG")


@app.route("/fast_video_uploads")
def fast_video_upload():
    content_path = request.json["content"]
    print(content_path, flush=True)
    if content_path == None:
        content_path = "fast_neural_style_pytorch/images/tokyo2.jpg"
    style_path = request.json["style"]
    styled_video_path = fast_style_transfer_video_file(content_path, style_path,)
    return styled_video_path


@click.command()
@click.option("--local/--remote", default=True)
@click.option("--lite/--no-lite", default=False)
def main(local, lite):
    if local:
        app.api_url = LOCAL_URL
    else:
        app.api_url = REMOTE_URL

    app.use_tflite = lite
    app.run(debug=False)  # change to False for actual deployment


if __name__ == "__main__":
    main()


def create_app():
    app.api_url = REMOTE_URL
    app.use_tflite = False
    return app
