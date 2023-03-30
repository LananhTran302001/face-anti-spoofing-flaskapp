import argparse
import os
import os.path as osp
from pathlib import Path
import cv2 as cv
import torch

from utils.load_data import LoadData
from utils.general import read_py_config
from utils.model import build_model
from utils.torchcnn import TorchCNN

from face_detect.haar_cascade_detector import HaarCascadeDetector
from face_detect.retina_face_detector import RetinaFaceDetector
from tqdm import tqdm

LWFAS = {
    "large": {"config": "configs/celeba-intra/large.py", "weights": "weights/v0/"},
    "large075": {"config": "configs/celeba-intra/large_075.py"},
    "small": {"config": "configs/celeba-intra/small.py"},
    "small_075": {"config": "configs/celeba-intra/small_075.py"}
}


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="antispoofing recognition live demo script"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="logs/small/MobileNet3_small.pth.tar",
        help="model path(s) for inference.",
    )
    parser.add_argument("--source", type=str, default=None, help="Input image or video")
    parser.add_argument("--webcam", action="store_true", help="whether to use webcam.")
    parser.add_argument(
        "--webcam-id",
        type=int,
        default=-1,
        help="The input web camera address, local camera or rtsp address.",
    )
    parser.add_argument(
        "--config", type=str, default=None, required=False, help="Configuration file"
    )
    parser.add_argument(
        "--face-detector", type=str, default="HCD", help="face detector"
    )
    parser.add_argument(
        "--spoof-thresh",
        type=float,
        default=0.8,
        help="Threshold for predicting spoof/real. The lower the more model oriented on spoofs",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--GPU", type=int, default=0, help="specify which GPU to use")
    parser.add_argument(
        "-l",
        "--cpu_extension",
        help="MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels "
        "impl.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="output",
        help="directory to save predictions in",
    )
    parser.add_argument(
        "--save-img", action="store_true", help="save inference results in images"
    )
    return parser.parse_args()


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


def infer_camera(spoof_model, face_detector, args):
    # define video capture
    cap = cv.VideoCapture(args.webcam_id)

    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # save_path = osp.join(args.save_dir, "record.mp4")
    recorder = cv.VideoWriter(
        "record.mp4", cv.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height)
    )

    while True:
        _, frame = cap.read()

        # Detect faces in image
        faces = face_detector.get_detections(frame)

        # Detect presentation attack
        spoof_conf = pred_spoof(frame, faces, spoof_model)

        # draw rectange bounding faces
        output_frame = draw_detections(frame, faces, spoof_conf, 0.7, show_conf=True)

        cv.imshow("hehe", output_frame)

        if args.save_img:
            recorder.write(output_frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if args.save_img:
        recorder.release()
    cv.destroyAllWindows()




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


def main(args):
    device = args.device + f":{args.GPU}" if args.device == "cuda" else "cpu"
    # print(device)

    # face detector
    if args.face_detector == "HCD":
        face_detector = HaarCascadeDetector(conf=0.5, expand_ratio=1.2)
    else:
        face_detector = RetinaFaceDetector(conf=0.5, expand_ratio=1.2)

    # build model
    config = read_py_config(args.config)
    model = build_model(config, device, strict=True, mode="eval")

    # load checkpoint
    checkpoint_path = os.path.join(
        config.checkpoint.experiment_path, config.checkpoint.snapshot_name
    )
    spoof_model = TorchCNN(model, checkpoint_path, config, device=device)

    if args.webcam:
        infer_camera(spoof_model=spoof_model, face_detector=face_detector, args=args)
    else:
        infer_source(spoof_model=spoof_model, face_detector=face_detector, args=args)

def face_detector(type="haar cascade"):
    if type == "haar cascade":
        return HaarCascadeDetector(conf=0.5, expand_ratio=1.2)
    elif type == "retina face":
        return RetinaFaceDetector(conf=0.5, expand_ratio=1.2)
    return None

def fas_model(type="large"):
    device = '0'
    config = read_py_config(LWFAS[type]["config"])
    weights = LWFAS[type]["weights"]
    model = build_model(config, device, strict=True, mode="eval")
    return TorchCNN(model=model, checkpoint_path=weights, config=config, device=device)
