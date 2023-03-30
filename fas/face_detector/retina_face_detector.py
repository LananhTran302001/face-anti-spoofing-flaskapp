import numpy as np
from retinaface import RetinaFace
from utils.general import clamp

class RetinaFaceDetector:
    """Wrapper class for face detector"""

    def __init__(self, conf=.6, expand_ratio=1.2):
        self.confidence = conf
        self.expand_ratio = expand_ratio


    def get_detections(self, frame):
        """Returns all detections on frame"""
        # retina face bounding boxes: left top right bottom (x1 y1 x2 y2)
        output = RetinaFace.detect_faces(frame)
        bboxes = np.zeros((len(output), 4), dtype=int)
        confs = np.zeros((len(output), 1), dtype=float)
        for i, face in enumerate(output.values()):
            bboxes[i] = face["facial_area"]
            confs[i] = face["score"]
        
        detections = self.__decode_detections(bboxes, confs, frame.shape)
        return detections
    

    def __decode_detections(self, bboxes, confs, frame_shape):
        """Decodes raw SSD output"""

        real_h, real_w, _ = frame_shape
        detections = []

        for i, box in enumerate(bboxes):
            if confs[i] > self.confidence:
                left = int(clamp(box[0], 0, real_w))
                top = int(clamp(box[1], 0, real_h))
                right = int(clamp(box[2], left, real_w))
                bottom = int(clamp(box[3], top, real_h))
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