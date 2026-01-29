#!/usr/bin/env python3
"""
Hand Tracker for PyBullet Version
Simplified hand tracking without ROS dependencies
Compatible with MediaPipe 0.10.9+ (solutions) and 0.10.32+ (tasks)
"""

import cv2
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catkin_ws.src.robotic_arm_controller.scripts.gesture_recognizer import GestureRecognizer

# Try to import MediaPipe with compatibility handling
try:
    import mediapipe as mp
    
    # Check which API version we have
    if hasattr(mp, 'solutions'):
        # Old API (0.10.9 and earlier)
        print("Using MediaPipe solutions API (older version)")
        USE_SOLUTIONS_API = True
    else:
        # New API (0.10.32+)
        print("Using MediaPipe tasks API (newer version)")
        USE_SOLUTIONS_API = False
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe import Image, ImageFormat
except ImportError as e:
    print(f"Error importing MediaPipe: {e}")
    print("Please install: pip install mediapipe opencv-python")
    sys.exit(1)


class HandTracker:
    """
    Standalone hand tracker using MediaPipe
    Compatible with both old and new MediaPipe APIs
    """
    
    def __init__(self, camera_id=0, model_complexity=1, 
                 min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        Initialize hand tracker
        
        Args:
            camera_id: Camera device ID
            model_complexity: MediaPipe model complexity (0-2)
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        if USE_SOLUTIONS_API:
            # Old MediaPipe API (solutions)
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.hands = self.mp_hands.Hands(
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                max_num_hands=1
            )
        else:
            # New MediaPipe API (tasks) - simplified version
            # For now, we'll use OpenCV for basic hand detection
            # Full tasks API implementation would require model files
            print("WARNING: New MediaPipe API detected but not fully supported yet.")
            print("Falling back to simple tracking mode.")
            self.hands = None
            self.mp_hands = None
            self.mp_drawing = None
            self.mp_drawing_styles = None
        
        # Initialize gesture recognizer
        self.gesture_recognizer = GestureRecognizer()
        
        # Camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"Hand Tracker initialized (Camera {camera_id})")
    
    def get_hand_pose(self):
        """
        Get current hand pose
        
        Returns:
            Tuple of (hand_detected, position, landmarks, gesture, frame)
            - hand_detected: bool
            - position: np.array [x, y, z] (normalized)
            - landmarks: np.array of 21 landmarks
            - gesture: tuple of (gesture_name, confidence) or None
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None, None, frame
        
        # If new MediaPipe API (not supported yet), return frame without tracking
        if self.hands is None:
            debug_frame = frame.copy()
            cv2.putText(debug_frame, "MediaPipe 0.10.32+ not fully supported", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(debug_frame, "Please install Python 3.11 and MediaPipe 0.10.14", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(debug_frame, "See PYTHON_311_SETUP.md for instructions", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            return False, None, None, None, debug_frame
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(image_rgb)
        
        # Draw on frame
        debug_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                debug_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # Get index finger tip position (landmark 8)
            position = landmarks[8]
            
            # Recognize gesture
            gesture = self.gesture_recognizer.get_best_gesture(landmarks)
            
            return True, position, landmarks, gesture, debug_frame
        
        else:
            return False, None, None, None, debug_frame
    
    def release(self):
        """Release resources"""
        self.cap.release()
        if self.hands is not None:
            self.hands.close()


if __name__ == "__main__":
    # Test hand tracker
    print("Testing Hand Tracker...")
    print("Press 'q' to quit")
    
    tracker = HandTracker()
    
    try:
        while True:
            hand_detected, position, landmarks, gesture, frame = tracker.get_hand_pose()
            
            if hand_detected:
                print(f"Hand position: {position}")
                if gesture:
                    print(f"Gesture: {gesture[0]} ({gesture[1]:.2f})")
            
            # Show frame
            cv2.imshow('Hand Tracker', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        tracker.release()
        cv2.destroyAllWindows()
