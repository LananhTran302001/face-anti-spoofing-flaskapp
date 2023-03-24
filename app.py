import os
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__, template_folder="template", static_folder="static")
app.secret_key = "abc"
app.config['UPLOAD_FOLDER'] = "static/upload"
app.config['UPLOAD_IMG_EXT'] = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
app.config['UPLOAD_VID_EXT'] = ["mp4", "mov", "avi", "mkv"]

input_img = "static/upload/creative.jpg"
model_cfg = None
model_weights = None
spoof_conf = None

def is_valid_file(file_path):
    extension = file_path.split('.')[-1].lower() 
    return extension in app.config['UPLOAD_IMG_EXT'] or extension in app.config['UPLOAD_VID_EXT']

@app.route('/')
def index():
    flash("Welcome to my website")
    return render_template("home.html")

@app.route('/success')
def success():
   return 'welcome la'

@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        input_file = request.files['input_file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], input_file.filename)
        input_file.save(path)
        return render_template('home.html', iimg=path)
    return render_template('home.html')

def get_input_file():
    pass

@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        return render_template('home.html', iimg=input_img, oimg=input_img)
    return redirect(url_for('success'))


if __name__ == '__main__':
   app.run(debug=True)
