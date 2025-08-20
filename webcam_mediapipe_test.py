import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Camera capture setup
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Try index 0 first
if not cap.isOpened():
    print("❌ Cannot open camera with index 0. Trying other indices.")
    for i in [-1, 1, 2]: # Try -1 for any available, then 1 and 2
        print(f"Trying camera index: {i}")
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"✅ Camera opened with index {i}")
            break
    if not cap.isOpened():
        print("❌ All attempts to open camera failed. Exiting.")
        exit()

print("Camera initialized. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Cannot grab frame. Exiting.")
        break

    # Flip the frame horizontally for a more natural view
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("MediaPipe Hand Tracking", frame)

    # Print mean pixel value (for debugging black screen issue)
    print("Mean pixel value:", frame.mean())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 