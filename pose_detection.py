import cv2
import mediapipe as mp
import numpy as np
import json
import time
from datetime import datetime

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure pose detection with higher accuracy
pose = mp_pose.Pose(
    min_detection_confidence=0.8,      # Increased from 0.5
    min_tracking_confidence=0.8,       # Increased from 0.5
    model_complexity=2,                # Increased from 1 for better accuracy
    smooth_landmarks=True,             # Enable smoothing
    enable_segmentation=False,         # Disable for performance
    smooth_segmentation=False
)

# Configure hand detection with higher accuracy
hands = mp_hands.Hands(
    min_detection_confidence=0.8,      # Increased from default
    min_tracking_confidence=0.8,       # Increased from default
    max_num_hands=2,                   # Detect up to 2 hands
    model_complexity=1,                # Use complex model for accuracy
    static_image_mode=False
)

# Camera capture setup with fallback indices
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Cannot open camera with index 0. Trying other indices.")
    for i in [-1, 1, 2]:
        print(f"Trying camera index: {i}")
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"✅ Camera opened with index {i}")
            break
    if not cap.isOpened():
        print("❌ All attempts to open camera failed. Exiting.")
        exit()

# Set camera properties for better performance and quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)    # Increased resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)    # Increased resolution
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)         # Enable autofocus
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)      # Set brightness to middle
cap.set(cv2.CAP_PROP_CONTRAST, 0.5)        # Set contrast to middle

print("Enhanced Pose & Hand Detection Started!")
print("Controls:")
print("  'q' - Quit")
print("  'r' - Reset data")
print("  's' - Save current session data")
print("  'c' - Capture single frame data")
print("  'v' - Toggle video recording")
print("  'h' - Toggle hand detection display")
print("  'p' - Toggle pose detection display")

# Data storage
pose_data = []
hand_data = []
frame_count = 0
start_time = time.time()
recording = False
show_hands = True
show_pose = True

# Video recording setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = None

def save_data_to_json():
    """Save all captured data to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pose_hand_data_{timestamp}.json"
    
    data = {
        "session_info": {
            "timestamp": timestamp,
            "duration_seconds": time.time() - start_time,
            "total_frames": frame_count,
            "pose_frames": len(pose_data),
            "hand_frames": len(hand_data)
        },
        "pose_data": pose_data,
        "hand_data": hand_data
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Data saved to {filename}")
    return filename

def capture_single_frame():
    """Capture data from current frame only"""
    if results_pose.pose_landmarks or results_hands.multi_hand_landmarks:
        frame_data = {
            "frame_number": frame_count,
            "timestamp": time.time() - start_time,
            "pose_landmarks": None,
            "hand_landmarks": None
        }
        
        if results_pose.pose_landmarks:
            landmarks = []
            for landmark in results_pose.pose_landmarks.landmark:
                landmarks.append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                })
            frame_data["pose_landmarks"] = landmarks
        
        if results_hands.multi_hand_landmarks:
            hands_landmarks = []
            for hand_landmarks in results_hands.multi_hand_landmarks:
                hand_data_single = []
                for landmark in hand_landmarks.landmark:
                    hand_data_single.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    })
                hands_landmarks.append(hand_data_single)
            frame_data["hand_landmarks"] = hands_landmarks
        
        # Save single frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"single_frame_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(frame_data, f, indent=2)
        
        print(f"✅ Single frame captured: {filename}")
        return frame_data
    
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Cannot grab frame. Exiting.")
        break

    # Flip frame horizontally for more intuitive view
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with both pose and hand detection
    results_pose = pose.process(rgb_frame)
    results_hands = hands.process(rgb_frame)
    
    # Draw pose landmarks if enabled
    if show_pose and results_pose.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Extract and store pose data
        landmarks = []
        for landmark in results_pose.pose_landmarks.landmark:
            landmarks.append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })
        
        pose_data.append({
            "frame": frame_count,
            "timestamp": time.time() - start_time,
            "landmarks": landmarks
        })
        
        # Display pose information
        cv2.putText(frame, f"Pose Detected! Frame: {frame_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show key pose points
        if len(landmarks) > 0:
            # Nose position
            nose = landmarks[0]
            cv2.putText(frame, f"Nose: ({nose['x']:.2f}, {nose['y']:.2f})", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Shoulders
            if len(landmarks) > 12:
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                cv2.putText(frame, f"L Shoulder: ({left_shoulder['x']:.2f}, {left_shoulder['y']:.2f})", 
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"R Shoulder: ({right_shoulder['x']:.2f}, {right_shoulder['y']:.2f})", 
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Draw hand landmarks if enabled
    if show_hands and results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Extract and store hand data
        hands_landmarks = []
        for hand_landmarks in results_hands.multi_hand_landmarks:
            hand_data_single = []
            for landmark in hand_landmarks.landmark:
                hand_data_single.append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                })
            hands_landmarks.append(hand_data_single)
        
        hand_data.append({
            "frame": frame_count,
            "timestamp": time.time() - start_time,
            "hands": hands_landmarks
        })
        
        # Display hand information
        cv2.putText(frame, f"Hands: {len(results_hands.multi_hand_landmarks)}", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display frame info
    cv2.putText(frame, f"Frames: {frame_count} | Pose: {len(pose_data)} | Hands: {len(hand_data)}", 
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display session duration
    duration = time.time() - start_time
    cv2.putText(frame, f"Duration: {duration:.1f}s", 
                (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display recording status
    if recording:
        cv2.putText(frame, "RECORDING", (10, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Debug: Print mean pixel value to check for black screen
    mean_pixel = frame.mean()
    cv2.putText(frame, f"Mean pixel: {mean_pixel:.2f}", (10, 210), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show the frame
    cv2.imshow("Enhanced Pose & Hand Detection", frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        pose_data = []
        hand_data = []
        frame_count = 0
        start_time = time.time()
        print("🔄 Data reset!")
    elif key == ord('s'):
        if pose_data or hand_data:
            filename = save_data_to_json()
        else:
            print("❌ No data to save!")
    elif key == ord('c'):
        capture_single_frame()
    elif key == ord('v'):
        if not recording:
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"pose_hand_video_{timestamp}.mp4"
            video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (1280, 720))
            recording = True
            print(f"🎥 Started recording: {video_filename}")
        else:
            # Stop recording
            if video_writer:
                video_writer.release()
                video_writer = None
            recording = False
            print("⏹️ Recording stopped!")
    elif key == ord('h'):
        show_hands = not show_hands
        print(f"👐 Hand detection display: {'ON' if show_hands else 'OFF'}")
    elif key == ord('p'):
        show_pose = not show_pose
        print(f"🧍 Pose detection display: {'ON' if show_pose else 'OFF'}")
    
    # Write frame to video if recording
    if recording and video_writer:
        video_writer.write(frame)
    
    frame_count += 1

# Cleanup
cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()

# Print final summary
print(f"\n📊 Session Summary:")
print(f"Total frames processed: {frame_count}")
print(f"Pose frames captured: {len(pose_data)}")
print(f"Hand frames captured: {len(hand_data)}")
print(f"Session duration: {time.time() - start_time:.2f} seconds")

if pose_data or hand_data:
    print(f"💾 Data can be saved using 's' key or automatically saved to JSON")
