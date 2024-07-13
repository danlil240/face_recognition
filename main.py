import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import cv2
import time
import logging
from database import FaceDatabase
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from utils import calculate_iou
import threading
import queue

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class VideoProcessor:

    def __init__(self, detector, recognizer, database):
        self.detector = detector
        self.recognizer = recognizer
        self.database = database
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.tracked_faces = []
        self.recognition_interval = 30
        self.processing_thread = None
        self.running = False

    def start_processing_thread(self):
        self.running = True
        self.processing_thread = threading.Thread(target=self.process_face_thread)
        self.processing_thread.start()

    def stop_processing_thread(self):
        self.running = False
        self.processing_queue.put((None, None))
        if self.processing_thread:
            self.processing_thread.join()

    def process_face_thread(self):
        while self.running:
            try:
                face, frame = self.processing_queue.get(timeout=1)
                if face is None and frame is None:
                    break

                x, y, w, h = face["box"]
                face_img = frame[y : y + h, x : x + w]
                face_img_bytes = face_img.tobytes()
                processed_embedding = self.recognizer.process_face(
                    face_img_bytes, face_img.shape
                )

                if processed_embedding is not None:
                    person_id, person_name = self.recognizer.recognize_person(
                        processed_embedding
                    )
                    self.result_queue.put((face, person_id, person_name))
                else:
                    self.result_queue.put((face, None, None))

                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in process_face_thread: {e}")

    def update_tracked_faces(self, detected_faces):
        new_tracked_faces = []
        for tracked_face in self.tracked_faces:
            matched = False
            for detected_face in detected_faces:
                if calculate_iou(detected_face["box"], tracked_face["box"]) > 0.5:
                    tracked_face.update(detected_face)
                    tracked_face["frames_since_detection"] = 0
                    new_tracked_faces.append(tracked_face)
                    matched = True
                    break
            if not matched:
                tracked_face["frames_since_detection"] += 1

        for detected_face in detected_faces:
            if not any(
                calculate_iou(detected_face["box"], face["box"]) > 0.5
                for face in new_tracked_faces
            ):
                new_tracked_faces.append(
                    {
                        **detected_face,
                        "id": None,
                        "name": None,
                        "frames_since_recognition": 0,
                        "frames_since_detection": 0,
                    }
                )

        self.tracked_faces = [
            face for face in new_tracked_faces if face["frames_since_detection"] < 10
        ]

    def process_results(self):
        while not self.result_queue.empty():
            face, person_id, person_name = self.result_queue.get()
            if person_id is not None:
                face["id"] = person_id
                face["name"] = person_name
            self.result_queue.task_done()


class FaceRecognitionApp:

    def __init__(self, database, detector, recognizer):
        self.database = database
        self.detector = detector
        self.recognizer = recognizer
        self.video_processor = VideoProcessor(detector, recognizer, database)
        self.selected_face = None
        self.input_text = ""

    def run(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Face Recognition")
        cv2.setMouseCallback("Face Recognition", self.mouse_callback)

        self.video_processor.start_processing_thread()

        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to capture frame")
                    break

                frame_count += 1
                if frame_count % 100 == 0:  # Check for merges every 100 frames
                    merged_count = self.recognizer.auto_merge_similar_persons()
                    logging.info(f"Performed {merged_count} merges")
                if frame_count % 5 == 0:
                    detected_faces = self.detector.detect_faces(frame)
                    self.video_processor.update_tracked_faces(detected_faces)

                for face in self.video_processor.tracked_faces:
                    self.draw_face_box(frame, face)
                    if (
                        face.get("id") is None
                        or face.get("frames_since_recognition", 0)
                        >= self.video_processor.recognition_interval
                    ):
                        self.video_processor.processing_queue.put((face, frame.copy()))
                        face["frames_since_recognition"] = 0
                    else:
                        face["frames_since_recognition"] = (
                            face.get("frames_since_recognition", 0) + 1
                        )

                self.video_processor.process_results()

                if self.selected_face:
                    self.draw_selected_face(frame)

                cv2.imshow("Face Recognition", frame)

                if self.handle_keyboard_input() == "quit":
                    break

        finally:
            self.video_processor.stop_processing_thread()
            cap.release()
            cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            logging.info("EVENT_LBUTTONDOWN")
            for face in self.video_processor.tracked_faces:
                fx, fy, fw, fh = face["box"]
                if fx < x < fx + fw and fy < y < fy + fh:
                    self.selected_face = face
                    break

    def draw_face_box(self, frame, face):
        x, y, w, h = face["box"]
        color = (0, 255, 0) if face != self.selected_face else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        name = face.get("name") or f"ID: {face.get('id', 'Unknown')}"
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def draw_selected_face(self, frame):
        x, y, w, h = self.selected_face["box"]
        cv2.rectangle(frame, (x, y - 30), (x + 200, y - 2), (255, 255, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(
            frame,
            f"Name: {self.input_text}",
            (x + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    def handle_keyboard_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return "quit"
        elif self.selected_face:
            if key == 13:  # Enter key
                if self.input_text:
                    self.selected_face["name"] = self.input_text
                    self.database.update_person_name(
                        self.selected_face["id"], self.input_text
                    )
                    self.recognizer.update_person_name(
                        self.selected_face["id"], self.input_text
                    )
                self.selected_face = None
                self.input_text = ""
            elif key == 27:  # Esc key
                self.selected_face = None
                self.input_text = ""
            elif key == 8:  # Backspace
                self.input_text = self.input_text[:-1]
            elif 32 <= key <= 126:  # Printable ASCII characters
                self.input_text += chr(key)
        return "continue"


def main():
    database = None
    try:
        database = FaceDatabase()
        detector = FaceDetector()
        recognizer = FaceRecognizer(database)

        app = FaceRecognitionApp(database, detector, recognizer)
        app.run()
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")


if __name__ == "__main__":
    main()
