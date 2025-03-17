import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize Mediapipe for hands and face detection
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hand and Face Detection
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize variables for FPS calculation
prev_time = 0
curr_time = 0

# Trail buffer for hand movement
trail_points = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Unable to capture video")
        break

    # Flip and convert the image to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hands and face
    hand_results = hands.process(rgb_frame)
    face_results = face_detection.process(rgb_frame)

    # Draw face detections
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            bbox = int(bbox.xmin * w), int(bbox.ymin * h), \
                   int(bbox.width * w), int(bbox.height * h)
            cv2.rectangle(frame, bbox, (255, 0, 0), 2)
            cv2.putText(frame, "Face Detected", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Draw hand landmarks and track hand movement
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the position of the index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            h, w, _ = frame.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Draw a circle around the fingertip
            cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)

            # Add trail points for hand movement
            trail_points.append((cx, cy))
            if len(trail_points) > 20:
                trail_points.pop(0)
            for point in trail_points:
                cv2.circle(frame, point, 5, (0, 255, 255), -1)

            # Calculate distance between thumb and index finger
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            distance = int(np.sqrt((thumb_x - cx)**2 + (thumb_y - cy)**2))
            cv2.line(frame, (cx, cy), (thumb_x, thumb_y), (255, 0, 0), 2)
            cv2.putText(frame, f"Distance: {distance}px", (cx, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Gesture recognition: Check for click (distance threshold)
            if distance < 30:
                cv2.putText(frame, "Click Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

    # Calculate and display FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Hand and Face Tracking with Features", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
