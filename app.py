import io
import json
import os

import click
import numpy as np
import requests
import tensorflow as tf
from flask import Flask, render_template, request, send_file
from PIL import Image
from werkzeug.utils import secure_filename

import style_video

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


def allowed_file(filename):
    return "." in filename and extension(filename) in ALLOWED_EXTENSIONS


def extension(filename):
    return filename.rsplit(".", 1)[1].lower()


@app.route("/upload_form", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file1" not in request.files or "file2" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file1 = request.files["file1"]
        file2 = request.files["file2"]
        # if user does not select file, browser also
        # submit an empty part without filename
        if file1.filename == "" or file2.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if (
            file1
            and allowed_file(file1.filename)
            and file2
            and allowed_file(file2.filename)
        ):
            filename1 = secure_filename(file1.filename)
            filepath1 = os.path.join(app.config["UPLOAD_FOLDER"], filename1)
            file1.save(filepath1)
            filename2 = secure_filename(file2.filename)
            filepath2 = os.path.join(app.config["UPLOAD_FOLDER"], filename2)
            file2.save(filepath2)
            data = {"content": filepath1, "style": filepath2, "lite": app.use_tflite}
            content_filetype = FILE_TYPE[extension(file1.filename)]
            if content_filetype == "video":
                r = requests.get(url=f"{app.api_url}/video_uploads", json=data)
                return render_template(
                    "video.html",
                    filename=r._content.decode("ascii"),
                    filetype=content_filetype,
                )
            else:
                r = requests.get(url=f"{app.api_url}/image_uploads", json=data)
                file_object = io.BytesIO(r._content)

                return send_file(file_object, mimetype="image/PNG")

    return render_template("upload_form.html")


@app.route("/video_uploads")
def video_upload():
    content_path = request.json["content"]
    style_path = request.json["style"]
    use_tflite = request.json["lite"]

    styled_video_path = style_video.style_transfer_video_file(
        content_path, style_path, use_tflite
    )

    return styled_video_path


@app.route("/image_uploads")
def image_upload():
    content_path = request.json["content"]
    style_path = request.json["style"]

    content_image = style_video.get_content_image_from_path(content_path)
    style_image = style_video.preprocesses_style_image(style_path)

    img = style_video.get_style_transfer(
        content_image, 0, style_image, use_tflite=app.use_tflite, send_image=True
    )

    file_object = io.BytesIO()
    img.save(file_object, "PNG")
    file_object.seek(0)

    return send_file(file_object, mimetype="image/PNG")
    

@click.command()
@click.option("--local/--remote", default=True)
@click.option("--lite/--no-lite", default=False)
def main(local, lite):
    if local:
        app.api_url = LOCAL_URL
    else:
        app.api_url = REMOTE_URL

    app.use_tflite = lite
    app.run(debug=False) # change to False for actual deployment


if __name__ == "__main__":
    main()


def create_app():
    return app
