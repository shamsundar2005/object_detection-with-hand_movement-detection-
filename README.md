# Real-Time Hand Position Detection with Object Classification

This project demonstrates the integration of real-time hand position detection using MediaPipe and object classification with MobileNetV2 in Python. It also includes text-to-speech feedback using the `pyttsx3` library, which announces when the user's right or left hand is raised or lowered.

## Features:
- Real-time hand position tracking using MediaPipe Pose.
- Object classification with MobileNetV2 pre-trained on ImageNet.
- Voice feedback using `pyttsx3` for when the right or left hand is raised or lowered.
- Asynchronous voice feedback to avoid blocking the main camera feed.

## Requirements:
- Python 3.x
- OpenCV
- TensorFlow
- MediaPipe
- `pyttsx3`
- `threading` (standard Python library)

## Installation:
1. **Install Python dependencies**:

   You can install the required libraries using `pip`:

   ```bash
   pip install opencv-python tensorflow mediapipe pyttsx3
   ```

2. **Install TensorFlow** (if not already installed):
   
   ```bash
   pip install tensorflow
   ```

3. **Install pyttsx3 for voice feedback**:

   ```bash
   pip install pyttsx3
   ```

## Usage:
1. Clone this repository or download the Python script.
2. Ensure your webcam is connected and accessible.
3. Run the Python script to start hand position detection and object classification.

   ```bash
   python hand_position_classification.py
   ```

4. When a hand is detected and raised or lowered, the system will announce it through the speaker using text-to-speech.

5. Press `ESC` to stop the program.

## Code Explanation:

- **Object Classification**:
   The `MobileNetV2` model is used for real-time object classification. The webcam feed is resized and preprocessed before being passed to the model for prediction. The top 3 predictions are displayed on the screen with their respective confidence scores.

- **Hand Position Detection**:
   The MediaPipe Pose solution tracks the landmarks for the shoulders and wrists to detect if the hands are raised or lowered. If the right or left hand is raised or lowered, the program provides voice feedback asynchronously to avoid interrupting the webcam feed.

- **Asynchronous Voice Feedback**:
   Using the `threading` module, the `pyttsx3` voice feedback is run in a separate thread, allowing the camera feed to continue without delays or stuttering.

## Troubleshooting:
- **Camera not detected**: Ensure the webcam is properly connected and accessible.
- **Performance issues**: If the frame rate drops, consider reducing the image resolution or optimizing the model usage for performance.
