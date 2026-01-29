#!/usr/bin/env python3
"""
Simple Demo - Test Kinematics and Hand Tracking Without PyBullet
This runs immediately while PyBullet installs
"""

import sys
import os
import cv2
import numpy as np

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
catkin_scripts = os.path.join(script_dir, 'catkin_ws', 'src', 'robotic_arm_controller', 'scripts')
sys.path.insert(0, catkin_scripts)

try:
    from kinematics import create_default_robot
    from gesture_recognizer import GestureRecognizer
    import mediapipe as mp
except ImportError as e:
    print(f"Error importing: {e}")
    print(f"\nSearched in: {catkin_scripts}")
    print("\nPlease install dependencies:")
    print("pip install mediapipe opencv-python numpy scipy")
    sys.exit(1)


class SimpleDemo:
    """Simple demo without PyBullet"""
    
    def __init__(self):
        print("=" * 60)
        print("Simple Demo - Hand Tracking + Kinematics")
        print("(PyBullet visualization will be available after installation)")
        print("=" * 60)
        
        # Initialize kinematics
        print("\n[1/3] Initializing kinematics...")
        self.robot = create_default_robot()
        
        # Initialize hand tracking
        print("[2/3] Initializing hand tracker...")
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=1
        )
        
        # Initialize gesture recognizer
        print("[3/3] Initializing gesture recognizer...")
        self.gesture_recognizer = GestureRecognizer()
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("ERROR: Could not open camera!")
            sys.exit(1)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n✓ Initialization complete!")
        print("\nControls:")
        print("  - Show your hand to see tracking")
        print("  - Try different gestures")
        print("  - Press 'q' to quit")
        print("\n" + "=" * 60 + "\n")
    
    def map_to_robot_workspace(self, hand_pos):
        """Map hand position to robot workspace"""
        # Simple mapping
        robot_x = 0.2 + (hand_pos[0] - 0.2) * 0.5  # 0.2 to 0.6
        robot_y = -0.3 + hand_pos[1] * 0.6  # -0.3 to 0.3
        robot_z = 0.1 + (hand_pos[2] + 0.1) * 2.0  # 0.1 to 0.5
        
        return np.array([robot_x, robot_y, robot_z])
    
    def run(self):
        """Main loop"""
        print("Starting demo...")
        print("Waiting for hand detection...\n")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Convert to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image_rgb)
                
                # Create display frame
                display = frame.copy()
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        display,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extract landmarks
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    
                    # Get index finger position
                    hand_pos = landmarks[8]
                    
                    # Map to robot workspace
                    robot_target = self.map_to_robot_workspace(hand_pos)
                    
                    # Solve IK
                    joint_angles = self.robot.inverse_kinematics_numerical(
                        robot_target,
                        method='jacobian'
                    )
                    
                    # Recognize gesture
                    gesture = self.gesture_recognizer.get_best_gesture(landmarks)
                    
                    # Display info
                    y_pos = 30
                    cv2.putText(display, "HAND DETECTED", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    y_pos += 30
                    cv2.putText(display, f"Hand: ({hand_pos[0]:.2f}, {hand_pos[1]:.2f}, {hand_pos[2]:.2f})",
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    y_pos += 25
                    cv2.putText(display, f"Robot Target: ({robot_target[0]:.2f}, {robot_target[1]:.2f}, {robot_target[2]:.2f})",
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    if joint_angles is not None:
                        y_pos += 25
                        cv2.putText(display, "IK: SUCCESS", (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        y_pos += 25
                        angles_str = f"Joints: [{', '.join([f'{a:.2f}' for a in joint_angles[:3]])}...]"
                        cv2.putText(display, angles_str, (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    else:
                        y_pos += 25
                        cv2.putText(display, "IK: OUT OF REACH", (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    if gesture:
                        y_pos += 30
                        cv2.putText(display, f"Gesture: {gesture[0].upper()} ({gesture[1]:.2f})",
                                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Print to console occasionally
                    if frame_count % 30 == 0:
                        status = "✓ IK OK" if joint_angles is not None else "✗ Out of reach"
                        gesture_str = f" | Gesture: {gesture[0]}" if gesture else ""
                        print(f"Target: ({robot_target[0]:.2f}, {robot_target[1]:.2f}, {robot_target[2]:.2f}) | {status}{gesture_str}")
                
                else:
                    cv2.putText(display, "NO HAND DETECTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(display, "Show your hand to the camera", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add instructions
                cv2.putText(display, "Press 'q' to quit", (10, display.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Show frame
                cv2.imshow('Hand Tracking Demo', display)
                
                frame_count += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\n\nCleaning up...")
        self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete!")


if __name__ == "__main__":
    try:
        demo = SimpleDemo()
        demo.run()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
