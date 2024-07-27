import numpy as np
from deepface import DeepFace
import cv2
import time
import utils
from face_recognition import FaceRecognizer
from database import FaceDatabase


# Define a function to get embeddings
database = FaceDatabase()
recognizer = FaceRecognizer(database)


# Function to measure the time taken for embeddings
def measure_time(img_path, iterations=10):
    times = []
    for _ in range(iterations):
        start_time = time.time()
        embedding, faces_area = recognizer.process_face(img_path)
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times), embedding, faces_area


# Path to the image
img_path = "images/image.png"

# Measure the average time for 10 iterations
average_time, embeddings, faces_area = measure_time(img_path)

# Print the average time
print(
    f"Average time taken for face embedding over 10 iterations: {average_time} seconds"
)

# Read image
frame = cv2.imread(img_path)

# Draw rectangles around all detected faces
for face_area in faces_area:
    frame = cv2.rectangle(
        frame,
        (face_area["x"], face_area["y"]),
        (face_area["x"] + face_area["w"], face_area["y"] + face_area["h"]),
        (0, 255, 0),
        2,
    )

# Display the image with rectangles
cv2.imshow("Image", frame)

# Wait for the user to press the "X" button or any key
while True:
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
