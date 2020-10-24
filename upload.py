import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

dirname = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(dirname, 'static/')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}
FILE_TYPE = {"jpg": "image", "jpeg": "image", "png": "image", "mp4": "video"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Copied from flask documentation
def allowed_file(filename):
    return '.' in filename and \
           extension(filename) in ALLOWED_EXTENSIONS

def extension(filename):
	return filename.rsplit('.', 1)[1].lower()

# @app.route('/success')
# def static():
# 	return '''
#     <!doctype html>
#     <p> Success! </p>
#     <title>Upload new File</title>
#     <h1>Upload new File</h1>
#     <form method=post enctype=multipart/form-data>
#       <input type=file name=file>
#       <input type=submit value=Upload>
#     </form>
#     '''

@app.route('/', methods=['GET', 'POST'])
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

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == "__main__":
	app.run()