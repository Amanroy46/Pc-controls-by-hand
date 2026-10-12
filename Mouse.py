import cv2
import mediapipe as mp
import pyautogui
import math
import subprocess

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize Webcam.
cap = cv2.VideoCapture(0)

# Get screen size.
screen_width, screen_height = pyautogui.size()

# Cursor speed factor.
cursor_speed = 1.0  # Adjust this value for smoother mouse movement.

# Disable PyAutoGUI fail-safe (optional, not recommended).
pyautogui.FAILSAFE = False  # Disable fail-safe to avoid exception.

# Smoothing factors.
previous_x, previous_y = 0, 0
smoothing_factor = 0.2  # Increase this for smoother transitions

# Function to calculate the distance between two points.
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def move_mouse_and_check_gestures(landmarks, frame_width, frame_height):
    global previous_x, previous_y
    
    # Get the landmarks for the fingers.
    index_finger_tip = landmarks[8]
    thumb_tip = landmarks[4]
    middle_finger_tip = landmarks[12]
    palm_landmark = landmarks[0]

    # Calculate distances for tap, pinch, and minimize actions.
    distance_to_palm = calculate_distance(index_finger_tip, palm_landmark)
    pinch_distance = calculate_distance(index_finger_tip, thumb_tip)
    minimize_distance = calculate_distance(thumb_tip, middle_finger_tip)

    # Thresholds for detecting gestures.
    tap_threshold = 0.04
    pinch_threshold = 0.02
    minimize_threshold = 0.02

    # Convert hand landmark coordinates to screen coordinates.
    target_x = int(index_finger_tip.x * screen_width * cursor_speed)
    target_y = int(index_finger_tip.y * screen_height * cursor_speed)

    # Ensure the mouse position stays within the screen boundaries.
    target_x = min(max(target_x, 0), screen_width - 1)
    target_y = min(max(target_y, 0), screen_height - 1)

    # Smooth movement.
    smoothed_x = previous_x + (target_x - previous_x) * smoothing_factor
    smoothed_y = previous_y + (target_y - previous_y) * smoothing_factor

    # Update previous position.
    previous_x, previous_y = smoothed_x, smoothed_y

    # Move the mouse cursor to the smoothed position.
    pyautogui.moveTo(smoothed_x, smoothed_y)

    # If a tap gesture is detected, simulate a left-click.
    if distance_to_palm < tap_threshold:
        pyautogui.click(button='left')

    # If a pinch gesture is detected, simulate a right-click.
    if pinch_distance < pinch_threshold:
        pyautogui.click(button='right')

    # If middle and thumb pinch is detected, minimize open applications.
    if minimize_distance < minimize_threshold:
        pyautogui.hotkey('win', 'd')  # Minimize all windows

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for natural hand movement.
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert the image to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hands.
    result = hands.process(rgb_frame)

    # If hands are found, draw landmarks and move the mouse.
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Move mouse cursor and check for gestures.
            move_mouse_and_check_gestures(hand_landmarks.landmark, frame_width, frame_height)

    # Display the frame.
    cv2.imshow("Hand Gesture Control", frame)

    # Break the loop if 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources.
cap.release()
cv2.destroyAllWindows()
