# Import necessary libraries
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np


# Load the face detector classifier and the pre-trained emotion detection model
face_classifier = cv2.CascadeClassifier(r'/home/azain/Documents/Companion-Robot-Project/haarcascade_frontalface_default.xml')
classifier = load_model(r'/home/azain/Documents/Companion-Robot-Project/model.h5')

# Define the list of emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize the webcam
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("/home/azain/Documents/Companion-Robot-Project/video.mp4")

# Loop to continuously get frames from the webcam
while True:
    # Read a frame from the webcam
    _, frame = cap.read()
    labels = []
    # Convert the frame to grayscale because the face detector requires gray images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_classifier.detectMultiScale(gray)

    # Iterate over each face found in the frame
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        # Extract the region of interest (ROI) in grayscale for emotion classification
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Proceed only if there's a face detected
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0  # Normalize the ROI
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict the emotion of the face
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]  # Get the label of the most probable emotion
            label_position = (x, y)
            # Display the label on the image
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Display "No Faces" when no faces are detected in the ROI
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Emotion Detector', frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()