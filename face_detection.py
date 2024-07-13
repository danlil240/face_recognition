import cv2
from mtcnn import MTCNN


class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_faces(self, frame):
        # Reduce frame size for faster processing
        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert to RGB (MTCNN uses RGB)
        rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces
        faces = self.detector.detect_faces(rgb_small_frame)
        # Scale up the face locations
        for face in faces:
            face["box"] = [coord for coord in face["box"]]
        return faces
