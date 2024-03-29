import cv2 as cv
from fas.utils.general import read_py_config
from fas.utils.model import build_model, time_synchronized
from fas.utils.torchcnn import TorchCNN

from fas.face_detector.haar_cascade_detector import HaarCascadeDetector
from fas.face_detector.retina_face_detector import RetinaFaceDetector

LWFAS = {
    "large": {
        "config": "fas/configs/celeba-intra/large.py",
        "weights": "fas/weights/celeba/MobileNet3_large.pth.tar",
    },
    "small": {
        "config": "fas/configs/celeba-intra/small.py",
        "weights": "fas/weights/celeba/MobileNet3_small.pth.tar",
    },
    "large_rf-f12": {
        "config": "fas/configs/celeba-intra/large.py",
        "weights": "fas/weights/celeba_rf/MobileNet3_large_rf-f12.pth.tar",
    },
    "large_rf-f12-e2": {
        "config": "fas/configs/celeba-intra/large.py",
        "weights": "fas/weights/celeba_rf/MobileNet3_large_rf-f12-e2.pth.tar",
    }
}


def face_detector(type="haar cascade"):
    if type == "haar cascade":
        return HaarCascadeDetector(conf=0.5, expand_ratio=1.2)
    elif type == "retina face":
        return RetinaFaceDetector(conf=0.5, expand_ratio=1.2)
    return None


def fas_model(type="large"):
    print("load model ", type)
    device = "cuda:0"
    config = read_py_config(LWFAS[type]["config"])
    weights = LWFAS[type]["weights"]
    model = build_model(config, device, strict=True, mode="eval")
    return TorchCNN(model=model, checkpoint_path=weights, config=config, device=device)


def pred_spoof(frame, detections, spoof_model):
    """Get prediction for all detected faces on the frame"""
    faces = []
    for rect in detections:
        # cut face according coordinates of detections
        # detections: [(bbox, conf), ..]
        faces.append(frame[rect[0][1] : rect[0][3], rect[0][0] : rect[0][2]])

    if faces:
        output = spoof_model.forward(faces)
        output = list(map(lambda x: x.reshape(-1), output))
        return output
    return None, None


def draw_detections(frame, detections, confidence, thresh, show_conf=False):
    """Draws detections and labels"""
    for i, rect in enumerate(detections):
        left, top, right, bottom = rect[0]
        if (
            confidence[i][1] > thresh
        ):  # if (spoof confidence > threshold) -> spoof face, else real face
            label = f"spoof: {round(confidence[i][1]*100, 3)}%"
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
        else:
            label = f"real: {round(confidence[i][0]*100, 3)}%"
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)
        # print("spoof conf = ", confidence[i][1], " live conf = ", confidence[i][0])
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
        top = max(top, label_size[1])
        if show_conf:
            cv.rectangle(
                frame,
                (left, top - label_size[1]),
                (left + label_size[0], top + base_line),
                (255, 255, 255),
                cv.FILLED,
            )
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    return frame


def infer_img(spoof_model, face_detector, img_path, save_path):

    fd_time = 0.0  
    fas_time = 0.0

    # read image
    img = cv.imread(img_path)

    # Detect faces in image
    t = time_synchronized()
    faces = face_detector.get_detections(img)
    fd_time += time_synchronized() - t

    # Detect presentation attack
    t = time_synchronized()
    spoof_conf = pred_spoof(img, faces, spoof_model)
    fas_time += time_synchronized() - t

    # draw rectange bounding faces
    out_img = draw_detections(img, faces, spoof_conf, 0.7, show_conf=True)

    # save output image
    cv.imwrite(save_path, out_img)

    # return time of face-detection and spoof-detection
    fd_time = round(fd_time * 1E3, 1)
    fas_time = round(fas_time * 1E3, 1)
    return fd_time, fas_time


def infer_frame(spoof_model, face_detector, frame):
    faces = face_detector.get_detections(frame)
    spoof_conf = pred_spoof(frame, faces, spoof_model)
    return draw_detections(frame, faces, spoof_conf, 0.7, show_conf=True)


def infer_video(spoof_model, face_detector, vid_path, save_path):
    # read video
    vid = cv.VideoCapture(vid_path)

    # video metadata
    fps = vid.get(cv.CAP_PROP_FPS)
    w = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

    # video output writer
    vid_writer = cv.VideoWriter(save_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # infer each frame in video
    success, frame = vid.read()
    while success:

        # Detect faces in image
        faces = face_detector.get_detections(frame)

        # Detect presentation attack
        spoof_conf = pred_spoof(frame, faces, spoof_model)

        # draw rectange bounding faces
        output_frame = draw_detections(frame, faces, spoof_conf, 0.7, show_conf=True)
        vid_writer.write(output_frame)

        # read the next frame
        success, frame = vid.read()
    
    # close the video writer
    vid_writer.release()

