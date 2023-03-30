import cv2
import numpy as np
from utils.general import clamp

class HaarCascadeDetector:
    """Wrapper class for face detector"""

    def __init__(self, conf=.6, expand_ratio=1.2, cascade_file="haarcascade_frontalface_default.xml"):
        self.confidence = conf
        self.expand_ratio = expand_ratio
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_file)

    def get_detections(self, frame):
        """Returns all detections on frame"""
        # retina face bounding boxes: left top right bottom (x1 y1 x2 y2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        output = self.face_cascade.detectMultiScale(gray, 1.3, 4)
        bboxes = np.zeros((len(output), 4), dtype=float)
        confs = np.zeros((len(output), 1), dtype=float)
        for i, face in enumerate(output):
            bboxes[i] = face
            confs[i] = 0.8
        
        detections = self.__decode_detections(bboxes, confs, frame.shape)
        return detections
    
    def __decode_detections(self, bboxes, confs, frame_shape):
        """Decodes raw SSD output"""

        real_h, real_w, _ = frame_shape
        detections = []

        for i, box in enumerate(bboxes):  # (box: x y w h)
            if confs[i] > self.confidence:
                left = int(clamp(box[0], 0, real_w))
                top = int(clamp(box[1], 0, real_h))
                right = int(clamp(box[0] + box[2], left, real_w))
                bottom = int(clamp(box[1] + box[3], top, real_h))
                
                # if expand crop
                if self.expand_ratio != 1:
                    w = (right - left)
                    h = (bottom - top)
                    dw = w * (self.expand_ratio - 1.) / 2
                    dh = h * (self.expand_ratio - 1.) / 2
                    left = int(clamp((left - dw), 0, real_w))
                    right = int(clamp((right + dw), 0, real_w))
                    top = int(clamp((top - dh), 0, real_h))
                    bottom = int(clamp((bottom + dh), 0, real_h))

                detections.append(((left, top, right, bottom), confs[i]))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)
        
        return detections