import numpy as np
import cv2
import time
import logging
from database import FaceDatabase
from face_recognition import FaceRecognizer
from video_processor import AdvancedVideoProcessor
from typing import List, Tuple, Any
from database_management_gui import DatabaseManagementGUI
from tracked_face import TrackedFace


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FaceRecognitionApp:
    def __init__(self, database, recognizer):
        self.database = database
        self.recognizer = recognizer
        self.video_processor = AdvancedVideoProcessor(recognizer, database)
        self.tracked_faces: List[TrackedFace] = []
        self.recognition_interval = 5
        self.max_prediction_frames = 10
        self.selected_face = None
        self.input_text = ""
        self.show_textbox = False
        self.db_gui = DatabaseManagementGUI(database, recognizer)
        self.refresh_data_interval = 20

    def run(self):
        cap = cv2.VideoCapture(0)
        self.video_processor.start_processing()
        cv2.namedWindow("Face Recognition")
        cv2.setMouseCallback("Face Recognition", self.mouse_callback)
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % self.refresh_data_interval == 0:
                    self.recognizer.refresh_data()
                if frame_count % self.recognition_interval == 0:
                    self.video_processor.add_frame(frame.copy())
                    results = self.video_processor.get_results()
                    self.update_tracked_faces(results)
                else:
                    for face in self.tracked_faces:
                        face.predict()
                self.draw_face_boxes(frame)
                self.draw_textbox(frame)

                cv2.imshow("Face Recognition", frame)
                if self.handle_keyboard_input() == "quit":
                    break
                frame_count += 1
        finally:
            self.video_processor.stop_processing()
            cap.release()
            cv2.destroyAllWindows()

    def update_tracked_faces(self, results: List[Tuple[Any, Any, Any]]):
        new_faces = []
        seen_ids = set()
        for face, person_id, person_name in results:
            if person_id in seen_ids:
                continue
            seen_ids.add(person_id)

            bbox = (face["x"], face["y"], face["w"], face["h"])
            for face in self.tracked_faces:
                if face.face_id == person_id:
                    face.update(bbox)
                    face.name = person_name
                    break
            else:
                logging.info(f"New face detected: {person_id}, {person_name}")
                face = TrackedFace(person_id, bbox, person_name)
                new_faces.append(face)

        self.tracked_faces.extend(new_faces)
        self.tracked_faces = [
            f for f in self.tracked_faces if f.last_seen <= self.max_prediction_frames
        ]

    def draw_face_boxes(self, frame):
        for face in self.tracked_faces:
            x, y, w, h = face.bbox
            color = (255, 0, 0) if self.selected_face else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"ID: {face.face_id}" if face.face_id else "Unknown"
            if face.name:
                label = face.name
            cv2.putText(
                frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )

    def draw_textbox(self, frame):
        if self.show_textbox and self.selected_face:
            x, y, w, h = self.selected_face.bbox
            textbox_x = x
            textbox_y = y - 30
            textbox_w = w
            textbox_h = 30
            cv2.rectangle(
                frame,
                (textbox_x, textbox_y),
                (textbox_x + textbox_w, textbox_y + textbox_h),
                (255, 255, 255),
                -1,
            )
            cv2.putText(
                frame,
                self.input_text,
                (textbox_x + 5, textbox_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for face in self.tracked_faces:
                fx, fy, fw, fh = face.bbox
                if fx < x < fx + fw and fy < y < fy + fh:
                    self.selected_face = face
                    self.show_textbox = True
                    break

    def handle_keyboard_input(self):
        key = cv2.waitKey(1) & 0xFF

        if self.selected_face:
            if key == 13:  # Enter key
                if self.input_text:
                    self.selected_face.name = self.input_text
                    self.database.update_person_name(
                        self.selected_face.face_id, self.input_text
                    )
                self.selected_face = None
                self.input_text = ""
                self.show_textbox = False
            elif key == 27:  # Esc key
                self.selected_face = None
                self.input_text = ""
                self.show_textbox = False
            elif key == 8:  # Backspace
                self.input_text = self.input_text[:-1]
            elif 32 <= key <= 126:  # Printable ASCII characters
                self.input_text += chr(key)
        elif key == ord("q"):
            return "quit"
        elif key == ord("d"):
            self.db_gui.start()
        return "continue"


def main():
    database = None
    try:
        database = FaceDatabase()
        recognizer = FaceRecognizer(database)
        time.sleep(6.0)
        app = FaceRecognitionApp(database, recognizer)
        app.run()
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")


if __name__ == "__main__":
    main()
