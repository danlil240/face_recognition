import time
import numpy as np


class TrackedFace:
    def __init__(self, face_id, bbox, name=None):
        self.face_id = face_id
        self.bbox = bbox
        self.name = name
        self.velocity = np.array([0, 0])
        self.last_seen = 0
        now = time.time()
        self.predict_t = now
        self.update_t = now
        self.last_center = np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2])

    def update(self, bbox):
        now = time.time()
        dt = now - self.update_t
        self.update_t = now
        self.predict_t = self.update_t
        new_center = np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2])

        self.velocity = (new_center - self.last_center) / dt
        self.bbox = bbox
        self.last_seen = 0
        self.last_center = new_center

    def predict(self):
        now = time.time()
        dt = now - self.predict_t
        self.predict_t = now
        center = np.array(
            [self.bbox[0] + self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2]
        )
        new_center = center + self.velocity * dt
        self.bbox = (
            int(new_center[0] - self.bbox[2] / 2),
            int(new_center[1] - self.bbox[3] / 2),
            self.bbox[2],
            self.bbox[3],
        )
        self.last_seen += 1
