import cv2
import numpy as np
from scipy.spatial.distance import cosine

# Load the reference image
reference_image_path = r"C:\Users\ksr20\OneDrive\Desktop\Kunal-Coding-programming stuff\Kunal-face-recognition-project-1\Code\Kunal-passport.jpg"
reference_image = cv2.imread(reference_image_path)
reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Load the Haar Cascade classifier for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detect faces in the reference image
faces_reference = face_classifier.detectMultiScale(reference_image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

# If no faces are found in the reference image, you can raise an error or handle it as needed

# Assuming you have only one face in the reference image, you can extract its coordinates and size
if len(faces_reference) != 1:
    raise ValueError("Exactly one face should be present in the reference image")

x_reference, y_reference, w_reference, h_reference = faces_reference[0]

# Extract the reference face region and convert it to grayscale
reference_face = reference_image_gray[y_reference : y_reference + h_reference, x_reference : x_reference + w_reference]

# Start capturing video from the webcam
video_capture = cv2.VideoCapture(0)
while True:
    ret, video_frame = video_capture.read()

    if not ret:
        break

    # Convert the video frame to grayscale
    gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the video frame
    faces_video = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    # If no faces are found in the video frame, continue to the next frame
    if len(faces_video) == 0:
        continue

    # # Assuming you have only one face in the video frame, you can extract its coordinates and size
    # if len(faces_video) != 1:
    #     continue

    x_video, y_video, w_video, h_video = faces_video[0]

    # Extract the video face region and convert it to grayscale
    video_face = gray_frame[y_video : y_video + h_video, x_video : x_video + w_video]

    # Resize the video face to match the shape of the reference face
    video_face_resized = cv2.resize(video_face, reference_face.shape[::-1])

    # Calculate the cosine similarity between the reference face and the resized video face
    similarity = 1 - cosine(reference_face.flatten(), video_face_resized.flatten())

    # Define a threshold for recognition
    recognition_threshold = 0.7  # Adjust as needed

    if similarity >= recognition_threshold:
        # Recognized face
        cv2.rectangle(video_frame, (x_video, y_video), (x_video + w_video, y_video + h_video), (0, 255, 0), 2)
        cv2.putText(video_frame, "Kunal", (x_video, y_video), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print("Recognized")
    else:
        # Unknown face
        cv2.rectangle(video_frame, (x_video, y_video), (x_video + w_video, y_video + h_video), (0, 0, 255), 2)
        cv2.putText(video_frame, "Unknown", (x_video, y_video), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the video frame
    cv2.imshow("Video", video_frame)

    # Wait for a key press to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()