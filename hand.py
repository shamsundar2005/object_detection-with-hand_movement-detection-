import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3  # Import the pyttsx3 library for text-to-speech
import threading  # Import the threading module to handle concurrent tasks
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Suppress unnecessary TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize pyttsx3 for voice feedback
engine = pyttsx3.init()

# Function to say text using pyttsx3
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to run speak in a separate thread
def speak_async(text):
    threading.Thread(target=speak, args=(text,)).start()

# Load the pre-trained MobileNetV2 model trained on ImageNet
try:
    model = MobileNetV2(weights='imagenet')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")

# Initialize OpenCV for webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Initialize MediaPipe for hand position detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Variables for tracking arm positions
landmarks_formed = False
right_hand_raised = False
left_hand_raised = False

# Start MediaPipe Pose for hand position tracking
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        print("Processing frame...")

        # Resize the frame to match MobileNetV2 input size (224x224) for classification
        img_resized = cv2.resize(frame, (224, 224))

        # Preprocess the image for MobileNetV2
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make predictions with MobileNetV2
        predictions = model.predict(img_array)

        # Decode the predictions into human-readable labels
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        print("Predictions: ", decoded_predictions)

        # Draw predictions on the frame
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            cv2.putText(frame, f"{label}: {score:.2f}", (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Process the pose landmarks
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # If landmarks can be formed (person detected)
        if results.pose_landmarks:
            if not landmarks_formed:
                landmarks_formed = True
                print("Person detected. Begin arm movements.")

            # Store important landmarks for right and left hand detection
            right_shoulder = results.pose_landmarks.landmark[12]
            right_wrist = results.pose_landmarks.landmark[16]
            left_shoulder = results.pose_landmarks.landmark[11]
            left_wrist = results.pose_landmarks.landmark[15]

            # Detect if right hand is raised or lowered
            if right_wrist.y < right_shoulder.y and not right_hand_raised:
                print("Right hand raised")
                speak_async("Right hand raised")  # Use async speak
                right_hand_raised = True
            elif right_wrist.y > right_shoulder.y and right_hand_raised:
                print("Right hand lowered")
                speak_async("Right hand lowered")  # Use async speak
                right_hand_raised = False

            # Detect if left hand is raised or lowered
            if left_wrist.y < left_shoulder.y and not left_hand_raised:
                print("Left hand raised")
                speak_async("Left hand raised")  # Use async speak
                left_hand_raised = True
            elif left_wrist.y > left_shoulder.y and left_hand_raised:
                print("Left hand lowered")
                speak_async("Left hand lowered")  # Use async speak
                left_hand_raised = False

        # If no person detected (no landmarks), indicate that
        elif landmarks_formed:
            landmarks_formed = False
            print("No person detected. Please re-enter the frame.")

        # Draw pose landmarks on the frame
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Show the frame with predictions and pose landmarks
        cv2.imshow("Real-Time Image Classification & Hand Position Detection", image_bgr)

        # Exit loop when 'ESC' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
