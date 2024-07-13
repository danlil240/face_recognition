import cv2
import time
import logging
from database import FaceDatabase
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from utils import calculate_iou

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

text_input_state = {"active": False, "text": "", "face_id": None, "position": None}

tracked_faces = []


def face_recognition_loop(database, detector, recognizer):
    global text_input_state
    cap = cv2.VideoCapture(0)
    tracked_faces = []
    recognition_interval = 30
    selected_face = None
    input_text = ""
    frame_count = 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_face
        if event == cv2.EVENT_LBUTTONDOWN:
            logging.info("EVENT_LBUTTONDOWN")
            for face in tracked_faces:
                fx, fy, fw, fh = face["box"]
                if fx < x < fx + fw and fy < y < fy + fh:
                    selected_face = face
                    break

    cv2.namedWindow("Face Recognition")
    cv2.setMouseCallback("Face Recognition", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame")
            break

        frame_count += 1
        if frame_count % 5 == 0:
            detected_faces = detector.detect_faces(frame)
            tracked_faces = update_tracked_faces(tracked_faces, detected_faces)

        for face in tracked_faces:
            x, y, w, h = face["box"]
            color = (0, 255, 0) if face != selected_face else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            name = face.get("name") or f"ID: {face.get('id', 'Unknown')}"
            cv2.putText(
                frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            process_face(face, frame, recognizer, database, recognition_interval)

        if selected_face:
            x, y, w, h = selected_face["box"]
            cv2.rectangle(frame, (x, y - 30), (x + 200, y - 2), (255, 255, 255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.putText(
                frame,
                f"Name: {input_text}",
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif selected_face:
            if key == 13:  # Enter key
                if input_text:
                    selected_face["name"] = input_text
                    database.update_person_name(selected_face["id"], input_text)
                    recognizer.update_person_name(selected_face["id"], input_text)
                selected_face = None
                input_text = ""
            elif key == 27:  # Esc key
                selected_face = None
                input_text = ""
            elif key == 8:  # Backspace
                input_text = input_text[:-1]
            elif 32 <= key <= 126:  # Printable ASCII characters
                input_text += chr(key)

    cap.release()
    cv2.destroyAllWindows()


def update_tracked_faces(tracked_faces, detected_faces):
    new_tracked_faces = []
    for tracked_face in tracked_faces:
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

    return [face for face in new_tracked_faces if face["frames_since_detection"] < 10]


def handle_name_update(face_id, new_name, database, recognizer, tracked_faces):
    database.update_person_name(face_id, new_name)
    recognizer.update_person_name(
        face_id, new_name
    )  # Add this method to FaceRecognizer
    for face in tracked_faces:
        if face["id"] == face_id:
            face["name"] = new_name
    logging.info(f"Updated name for face {face_id} to {new_name}")


def process_face(face, frame, recognizer, database, recognition_interval):
    x, y, w, h = face["box"]
    face["frames_since_recognition"] += 1

    if face["id"] is None or face["frames_since_recognition"] >= recognition_interval:
        face_img = frame[y : y + h, x : x + w]
        face_img_bytes = face_img.tobytes()
        processed_embedding = recognizer.process_face(face_img_bytes, face_img.shape)
        if processed_embedding is not None:
            person_id, person_name = recognizer.recognize_person(processed_embedding)
            face["id"] = person_id
            face["frames_since_recognition"] = 0

            if person_id is not None:
                if recognizer.real_time_merge(person_id, processed_embedding):
                    person_name = database.get_person_name(person_id)
                # Always update the name from the database
                face["name"] = database.get_person_name(person_id)


def manage_database(database, recognizer):
    while True:
        print("\nDatabase Management:")
        print("1. List all persons")
        print("2. Update person's name")
        print("3. Delete person")
        print("4. Auto-merge similar persons")
        print("5. Manual merge")
        print("6. Back to main menu")
        choice = input("Enter your choice: ")

        if choice == "1":
            persons = database.get_all_persons()
            for person in persons:
                print(f"ID: {person[0]}, Name: {person[1] or 'Unnamed'}")
        elif choice == "2":
            id = input("Enter person's ID: ")
            name = input("Enter new name: ")
            database.update_person_name(id, name)
            print("Name updated successfully")
        elif choice == "3":
            try:
                removed_count = database.clean_database()
                print(f"Database cleaned. Removed {removed_count} invalid entries.")
            except Exception as e:
                print(f"An error occurred while cleaning the database: {e}")
        elif choice == "4":
            merged_count = recognizer.auto_merge_similar_persons()
            print(f"Auto-merged {merged_count} pairs of similar persons")
        elif choice == "5":
            id1 = input("Enter ID of the first person: ")
            id2 = input("Enter ID of the second person: ")
            database.merge_persons(int(id1), int(id2))
            print(f"Merged person {id2} into {id1}")
        elif choice == "6":
            break
        else:
            print("Invalid choice. Please try again.")


def main():
    database = FaceDatabase()
    detector = FaceDetector()
    recognizer = FaceRecognizer(database)

    while True:
        print("\nMain Menu:")
        print("1. Start Face Recognition")
        print("2. Manage Database")
        print("3. Clean Database (remove incorrect embeddings)")
        print("4. Reset Database (clear all data)")
        print("5. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            face_recognition_loop(database, detector, recognizer)
        elif choice == "2":
            manage_database(database, recognizer)
        elif choice == "3":
            removed_count = database.clean_database()
            print(f"Database cleaned. Removed {removed_count} invalid entries.")
        elif choice == "4":
            confirm = input(
                "Are you sure you want to reset the database? This will erase all data and reset IDs. (y/n): "
            )
            if confirm.lower() == "y":
                if database.reset_database():
                    print(
                        "Database reset. All data has been erased and IDs have been reset."
                    )
                else:
                    print(
                        "An error occurred while resetting the database. Please check the logs."
                    )
            else:
                print("Database reset cancelled.")
        elif choice == "5":
            break
        else:
            print("Invalid choice. Please try again.")

    database.close()


if __name__ == "__main__":
    main()
