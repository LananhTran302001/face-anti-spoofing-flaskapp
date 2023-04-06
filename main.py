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
)
from fas.inferer import face_detector, fas_model, infer_img, infer_video

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
    global fd, fas
    fd = face_detector(FACE_DETECTORS[0])
    fas = fas_model(FAS_MODELS[0])
    session["face_detector"] = FACE_DETECTORS[0]
    session["fas_model"] = FAS_MODELS[0]
    return render_template(
        "home.html",
        face_detectors=FACE_DETECTORS,
        fas_models=FAS_MODELS,
        selected_face_detector=session["face_detector"],
        selected_fas_model=session["fas_model"],
        on_camera=False,
    )


@app.route("/", methods=["POST", "GET"])
def upload_file():
    if request.method == "POST":
        input_file = request.files["input_file"]

        if is_image(input_file.filename):
            path = os.path.join(app.config["UPLOAD_FOLDER"], input_file.filename)
            input_file.save(path)
            session["uploaded_img_path"] = path
            return render_template(
                "home.html",
                iimg=path,
                face_detectors=FACE_DETECTORS,
                fas_models=FAS_MODELS,
                on_camera=False,
            )
        elif is_video(input_file.filename):
            path = os.path.join(app.config["UPLOAD_FOLDER"], input_file.filename)
            input_file.save(path)
            session["uploaded_img_path"] = path
            return render_template(
                "home.html",
                ivideo=path,
                face_detectors=FACE_DETECTORS,
                fas_models=FAS_MODELS,
                on_camera=False,
            )
        else:
            msg = "Your upload file must be an image or a video"
            return render_template(
                "home.html",
                face_detectors=FACE_DETECTORS,
                fas_models=FAS_MODELS,
                on_camera=False,
                message=msg,
            )

    return render_template(
        "home.html",
        face_detectors=FACE_DETECTORS,
        fas_models=FAS_MODELS,
        on_camera=False,
    )


@app.route("/camera")
def camera():
    global cap
    cap = cv2.VideoCapture(0)
    return render_template(
        "home.html",
        face_detectors=FACE_DETECTORS,
        fas_models=FAS_MODELS,
        on_camera=True,
    )


@app.route("/stream", methods=["GET"])
def stream():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/submit", methods=["POST", "GET"])
def submit():
    global fd, fas
    if request.method == "POST":
        if request.form.get("submit") == "Submit":
            if session["fas_model"] != request.form.get("fas-model-btn"):
                session["fas_model"] = request.form.get("fas-model-btn")
                fas = fas_model(session["fas_model"])

            if session["face_detector"] != request.form.get("face-detector-btn"):
                session["face_detector"] = request.form.get("face-detector-btn")
                fd = face_detector(session["face_detector"])

            output_path = os.path.join(
                app.config["OUTPUT_FOLDER"],
                os.path.basename(session["uploaded_img_path"]),
            )

            if is_image(session["uploaded_img_path"]):
                infer_img(
                    spoof_model=fas,
                    face_detector=fd,
                    img_path=session["uploaded_img_path"],
                    save_path=output_path,
                )
                session["last_output_img"] = output_path
                return render_template(
                    "home.html",
                    face_detectors=FACE_DETECTORS,
                    fas_models=FAS_MODELS,
                    on_camera=False,
                    iimg=session["uploaded_img_path"],
                    oimg=session["last_output_img"],
                    selected_face_detector=session["face_detector"],
                    selected_fas_model=session["fas_model"],
                )

            elif is_video(session["uploaded_img_path"]):
                infer_video(
                    spoof_model=fas,
                    face_detector=fd,
                    vid_path=session["uploaded_img_path"],
                    save_path=output_path,
                )
                session["last_output_img"] = output_path
                return render_template(
                    "home.html",
                    face_detectors=FACE_DETECTORS,
                    fas_models=FAS_MODELS,
                    on_camera=False,
                    ivideo=session["uploaded_img_path"],
                    ovideo=session["last_output_img"],
                    selected_face_detector=session["face_detector"],
                    selected_fas_model=session["fas_model"],
                )
        # elif request.form.get("start") == "Start":

        elif request.form.get("stop") == "Stop":
            cap.release()
            cv2.destroyAllWindows()

    return redirect("/")


@app.route("/download", methods=["GET"])
def download():
    return send_file(session["last_output_img"], as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
