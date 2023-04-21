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
    url_for,
)
from fas.inferer import face_detector, fas_model, infer_img, infer_video, infer_frame

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
FAS_MODELS = ["large", "small", "large_rf-f12", "large_rf-f12-e2"]

global cap, fd, fas, cam_on

cam_on = False
cap = None
fd = None
fas = None


def get_media_file(filename):
    return os.path.join(app.config["UPLOAD_FOLDER"], filename)


def is_image(file_path):
    extension = file_path.split(".")[-1].lower()
    return extension in app.config["UPLOAD_IMG_EXT"]


def is_video(file_path):
    extension = file_path.split(".")[-1].lower()
    return extension in app.config["UPLOAD_VID_EXT"]


def render_upload(
    html="upload_file.html",
    iimg=None,
    oimg=None,
    ivideo=None,
    ovideo=None,
    face_detectors=FACE_DETECTORS,
    fas_models=FAS_MODELS,
    selected_face_detector=FACE_DETECTORS[0],
    selected_fas_model=FAS_MODELS[0],
    fd_time=None,
    fas_time=None,
    noti=None
):
    return render_template(
        html,
        iimg=iimg,
        oimg=oimg,
        ivideo=ivideo,
        ovideo=ovideo,
        face_detectors=face_detectors,
        fas_models=fas_models,
        selected_face_detector=selected_face_detector,
        selected_fas_model=selected_fas_model,
        fd_time=fd_time,
        fas_time=fas_time,
        noti=noti
    )


def render_camera(
    html="camera.html",
    face_detectors=FACE_DETECTORS,
    fas_models=FAS_MODELS,
    selected_face_detector=FACE_DETECTORS[0],
    selected_fas_model=FAS_MODELS[0],
    noti=None
):
    global cam_on
    return render_template(
        html,
        cam_on=cam_on,
        face_detectors=face_detectors,
        fas_models=fas_models,
        selected_face_detector=selected_face_detector,
        selected_fas_model=selected_fas_model,
        noti = noti
    )


def render_phonecamera(
    html="phone_camera.html",
    cam_ip=None,
    face_detectors=FACE_DETECTORS,
    fas_models=FAS_MODELS,
    selected_face_detector=FACE_DETECTORS[0],
    selected_fas_model=FAS_MODELS[0],
    noti=None
):
    global cam_on
    return render_template(
        html,
        cam_on=cam_on,
        cam_ip=cam_ip,
        face_detectors=face_detectors,
        fas_models=fas_models,
        selected_face_detector=selected_face_detector,
        selected_fas_model=selected_fas_model,
        noti = noti
    )


def generate_frames():
    global fd, fas
    while True:
        ## read the camera frame
        success, frame = cap.read()
        if not success:
            print("Not success")
            break
        else:
            # detect spoofing face
            out_frame = infer_frame(spoof_model=fas, face_detector=fd, frame=frame)
            _, buffer = cv2.imencode(".jpg", out_frame)
            out_frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + out_frame + b"\r\n")


@app.route("/")
def index():
    session["fas_model"] = FAS_MODELS[0]
    session["face_detector"] = FACE_DETECTORS[0]
    return render_template(
        "home.html",
        face_detectors=FACE_DETECTORS,
        fas_models=FAS_MODELS,
    )


@app.route("/", methods=["POST"])
def goto():
    if request.form.get("upload") == "Upload":
        return redirect(url_for("upload"))
    elif request.form.get("camera") == "Camera":
        return redirect(url_for("camera"))
    elif request.form.get("mobile-phone-camera") == "Mobile phone camera":
        return redirect(url_for("phonecamera"))
    return redirect(url_for("index"))


@app.route("/back", methods=["GET"])
def backtohome():
    global cap, cam_on
    if cam_on:
        cap.release()
        cam_on = False
    return redirect(url_for("index"))


@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == "POST":
        input_file = request.files["input_file"]

        if is_image(input_file.filename):
            path = get_media_file(input_file.filename)
            input_file.save(path)
            session["uploaded_img_path"] = path
            return render_upload(iimg=path)

        elif is_video(input_file.filename):
            path = get_media_file(input_file.filename)
            input_file.save(path)
            session["uploaded_img_path"] = path
            return render_upload(ivideo=path)

        else:
            return render_upload(noti="Please upload image or video file")

    return render_upload()


@app.route("/camera", methods=["GET", "POST"])
def camera():
    global cap, cam_on, fas, fd
    if request.method == "GET":
        session["fas_model"] = FAS_MODELS[0]
        session["face_detector"] = FACE_DETECTORS[0]
        if cam_on:
            cap.release()
            cam_on = False
    else:
        if request.form.get("start") == "Start":
            if (not fas) or (session["fas_model"] != request.form.get("fas-model-btn")):
                session["fas_model"] = request.form.get("fas-model-btn")
                fas = fas_model(session["fas_model"])
            if (not fd) or (
                session["face_detector"] != request.form.get("face-detector-btn")
            ):
                session["face_detector"] = request.form.get("face-detector-btn")
                fd = face_detector(session["face_detector"])
            cam_on = True
            cap = cv2.VideoCapture(0)

        elif request.form.get("stop") == "Stop":
            cap.release()
            cam_on = False

    return render_camera()


@app.route("/phonecamera", methods=["GET", "POST"])
def phonecamera():
    global cap, cam_on, fd, fas
    if request.method == "GET":
        session["fas_model"] = FAS_MODELS[0]
        session["face_detector"] = FACE_DETECTORS[0]
        if cam_on:
            cap.release()
            cam_on = False
    else:
        if request.form.get("start") == "Start":
            if (not fas) or (session["fas_model"] != request.form.get("fas-model-btn")):
                session["fas_model"] = request.form.get("fas-model-btn")
                fas = fas_model(session["fas_model"])
            if (not fd) or (
                session["face_detector"] != request.form.get("face-detector-btn")
            ):
                session["face_detector"] = request.form.get("face-detector-btn")
                fd = face_detector(session["face_detector"])

            cam_ip = request.form.get("cam_ip")
            cap = cv2.VideoCapture("https://" + cam_ip + "/video")
            cam_on = True
            return render_phonecamera(cam_ip=cam_ip)

        elif request.form.get("stop") == "Stop":
            cap.release()
            cam_on = False
    return render_phonecamera()


@app.route("/stream", methods=["GET"])
def stream():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/submit", methods=["POST", "GET"])
def submit():
    global fd, fas
    if request.method == "POST":
        if (not fas) or (session["fas_model"] != request.form.get("fas-model-btn")):
            session["fas_model"] = request.form.get("fas-model-btn")
            fas = fas_model(session["fas_model"])
        if (not fd) or (
            session["face_detector"] != request.form.get("face-detector-btn")
        ):
            session["face_detector"] = request.form.get("face-detector-btn")
            fd = face_detector(session["face_detector"])
        
        output_path = os.path.join(
            app.config["OUTPUT_FOLDER"],
            os.path.basename(session["uploaded_img_path"]),
        )

        if is_image(session["uploaded_img_path"]):
            fd_time, fas_time = infer_img(
                spoof_model=fas,
                face_detector=fd,
                img_path=session["uploaded_img_path"],
                save_path=output_path,
            )
            session["last_output_img"] = output_path
            return render_upload(
                selected_face_detector=session["face_detector"],
                selected_fas_model=session["fas_model"],
                iimg=session["uploaded_img_path"],
                oimg=session["last_output_img"],
                fd_time=fd_time,
                fas_time=fas_time,
            )

        elif is_video(session["uploaded_img_path"]):
            infer_video(
                spoof_model=fas,
                face_detector=fd,
                vid_path=session["uploaded_img_path"],
                save_path=output_path,
            )
            session["last_output_img"] = output_path
            return render_upload(
                ivideo=session["uploaded_img_path"],
                ovideo=session["last_output_img"],
                selected_face_detector=session["face_detector"],
                selected_fas_model=session["fas_model"],
            )
        else:
            return render_upload(
                iimg=session["uploaded_img_path"],
                oimg=session["last_output_img"],
                selected_face_detector=session["face_detector"],
                selected_fas_model=session["fas_model"],
            )
        # elif request.form.get("start") == "Start":

    return redirect("/")


@app.route("/download", methods=["GET"])
def download():
    return send_file(session["last_output_img"], as_attachment=True)


if __name__ == "__main__":
    app.run()
