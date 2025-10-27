import dlib
import cv2
from math import hypot
import numpy as np

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Start webcam feed
cap = cv2.VideoCapture(0)

def midpoint(p1, p2):
    """Returns the midpoint between two dlib points."""
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_blinking_ratio(eye_points, facial_landmarks):
    """Calculates the blinking ratio (horizontal/vertical eye aspect)."""
    left_point = facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y
    right_point = facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y

    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = hypot(left_point[0] - right_point[0], left_point[1] - right_point[1])
    ver_line_length = hypot(center_top[0] - center_bottom[0], center_top[1] - center_bottom[1])

    return hor_line_length / ver_line_length

MAX_THRESH = 20   # Number of consecutive closed-eye frames before alert
count = 0         # Keeps track of how long eyes remain closed

while True:
    success, img = cap.read()
    if not success:
        break

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)

    for face in faces:
        landmarks = predictor(imgGray, face)

        # Get eye aspect ratios
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # Detect blink or prolonged eye closure
        if blinking_ratio > 5.0:
            cv2.putText(img, "Blinking", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            count += 1
            if count >= MAX_THRESH:
                cv2.putText(img, "Drowsiness Detected", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        else:
            count = 0

    # Display live feed
    cv2.imshow('Facial Landmark Detection', img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
