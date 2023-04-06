import os
import cv2
from flask import (
    Flask,
    Response,
    render_template,
    request,
    session,
    redirect,
    send_file,
    url_for
)
# from fas.inferer import face_detector, fas_model, infer_img, infer_video

app = Flask(__name__, template_folder="template", static_folder="static")
app.secret_key = "abc"
app.config["UPLOAD_FOLDER"] = "static/upload"
app.config["UPLOAD_IMG_EXT"] = [
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "dng",
    "webp",
    "mpo",
]
app.config["UPLOAD_VID_EXT"] = ["mp4", "mov", "avi", "mkv"]
app.config["OUTPUT_FOLDER"] = "static/output"

FACE_DETECTORS = ["haar cascade", "retina face"]
FAS_MODELS = ["large", "small", "large075", "small075"]

global cap, fd, fas
cap = cv2.VideoCapture(0)


def is_image(file_path):
    extension = file_path.split(".")[-1].lower()
    return extension in app.config["UPLOAD_IMG_EXT"]


def is_video(file_path):
    extension = file_path.split(".")[-1].lower()
    return extension in app.config["UPLOAD_VID_EXT"]


def generate_frames():
    while True:
        ## read the camera frame
        success, frame = cap.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template(
        "home.html"
    )


@app.route("/", methods=["POST", "GET"])
def goto():
    if request.form.get("upload") == "Upload":
        return redirect(url_for("upload"))
    elif request.form.get("camera") == "Camera":
        return redirect(url_for("stream"))
    return redirect(url_for("index"))

@app.route("/back", methods=["GET"])
def backtohome():
    return redirect(url_for("index"))


@app.route("/upload", methods=["POST", "GET"])
def upload_file():
    if request.method == "POST":
        input_file = request.files["input_file"]

        if is_image(input_file.filename):
            path = os.path.join(app.config["UPLOAD_FOLDER"], input_file.filename)
            input_file.save(path)
            session["uploaded_img_path"] = path
            return render_template(
                "upload_file.html",
                iimg=path,
                face_detectors=FACE_DETECTORS,
                fas_models=FAS_MODELS
            )
        elif is_video(input_file.filename):
            path = os.path.join(app.config["UPLOAD_FOLDER"], input_file.filename)
            input_file.save(path)
            session["uploaded_img_path"] = path
            return render_template(
                "upload_file.html",
                ivideo=path,
                face_detectors=FACE_DETECTORS,
                fas_models=FAS_MODELS
            )
        else:
            msg = "Your upload file must be an image or a video"
            return render_template(
                "upload_file.html",
                face_detectors=FACE_DETECTORS,
                fas_models=FAS_MODELS,
                message=msg
            )

    return render_template("upload_file.html")




@app.route("/stream", methods=["POST", "GET"])
def stream():
    if request.method == "POST":
        return render_template("camera.html")
    else:
        global cap
        cap = cv2.VideoCapture(0)
        return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == '__main__':  
   app.run()