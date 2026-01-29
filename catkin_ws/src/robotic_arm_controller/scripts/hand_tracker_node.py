#!/usr/bin/env python3
"""
Hand Tracker ROS Node
Uses MediaPipe to track hand position and gestures from camera feed
Publishes hand pose and gesture commands to ROS topics
"""

import rospy
import cv2
import mediapipe as mp
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from robotic_arm_controller.msg import HandPose, GestureCommand
from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import Header

# Import local modules
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from gesture_recognizer import GestureRecognizer
from transform_utils import ExponentialFilter


class HandTrackerNode:
    """
    ROS node for hand tracking using MediaPipe
    """
    
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('hand_tracker_node', anonymous=True)
        
        # Parameters
        self.camera_id = rospy.get_param('~camera_id', 0)
        self.model_complexity = rospy.get_param('~model_complexity', 1)
        self.min_detection_confidence = rospy.get_param('~min_detection_confidence', 0.7)
        self.min_tracking_confidence = rospy.get_param('~min_tracking_confidence', 0.5)
        self.smoothing_alpha = rospy.get_param('~smoothing_alpha', 0.3)
        self.publish_debug_image = rospy.get_param('~publish_debug_image', True)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            max_num_hands=1  # Track only one hand
        )
        
        # Initialize gesture recognizer
        self.gesture_recognizer = GestureRecognizer(
            confidence_threshold=0.7,
            debounce_time=0.5
        )
        
        # Initialize filters for smoothing
        self.position_filter = ExponentialFilter(alpha=self.smoothing_alpha)
        
        # Publishers
        self.hand_pose_pub = rospy.Publisher('/hand_pose', HandPose, queue_size=10)
        self.gesture_pub = rospy.Publisher('/gesture_command', GestureCommand, queue_size=10)
        
        if self.publish_debug_image:
            self.debug_image_pub = rospy.Publisher('/hand_tracker/debug_image', Image, queue_size=1)
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Camera capture
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            rospy.logerr(f"Failed to open camera {self.camera_id}")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        rospy.loginfo("Hand Tracker Node initialized")
        rospy.loginfo(f"Camera ID: {self.camera_id}")
        rospy.loginfo(f"Model complexity: {self.model_complexity}")
        rospy.loginfo(f"Detection confidence: {self.min_detection_confidence}")
    
    def process_frame(self, frame):
        """
        Process a single frame to detect hand and extract pose
        
        Args:
            frame: OpenCV image (BGR)
            
        Returns:
            Tuple of (hand_detected, hand_pose_msg, gesture_msg, debug_image)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(image_rgb)
        
        # Create debug image
        debug_image = frame.copy() if self.publish_debug_image else None
        
        if results.multi_hand_landmarks:
            # Get first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks on debug image
            if debug_image is not None:
                self.mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Extract landmarks as numpy array
            landmarks = self.extract_landmarks(hand_landmarks, frame.shape)
            
            # Calculate hand pose (using wrist and middle finger MCP)
            wrist = landmarks[0]
            middle_mcp = landmarks[9]
            index_tip = landmarks[8]
            
            # Hand position (use index fingertip as control point)
            hand_position = index_tip
            
            # Smooth position
            smoothed_position = self.position_filter.update(hand_position)
            
            # Create hand pose message
            hand_pose_msg = HandPose()
            hand_pose_msg.header = Header()
            hand_pose_msg.header.stamp = rospy.Time.now()
            hand_pose_msg.header.frame_id = "camera_frame"
            
            hand_pose_msg.position = Point(
                x=smoothed_position[0],
                y=smoothed_position[1],
                z=smoothed_position[2]
            )
            
            # Simple orientation (can be improved)
            hand_pose_msg.orientation = Quaternion(x=0, y=0, z=0, w=1)
            hand_pose_msg.confidence = 1.0
            hand_pose_msg.hand_detected = True
            
            # Recognize gestures
            gesture_result = self.gesture_recognizer.get_best_gesture(landmarks)
            
            gesture_msg = None
            if gesture_result is not None:
                gesture_name, gesture_conf = gesture_result
                
                gesture_msg = GestureCommand()
                gesture_msg.header = Header()
                gesture_msg.header.stamp = rospy.Time.now()
                gesture_msg.gesture_type = gesture_name
                gesture_msg.confidence = gesture_conf
                
                # Draw gesture on debug image
                if debug_image is not None:
                    cv2.putText(
                        debug_image,
                        f"Gesture: {gesture_name} ({gesture_conf:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
            
            # Draw position on debug image
            if debug_image is not None:
                pos_text = f"Pos: ({smoothed_position[0]:.2f}, {smoothed_position[1]:.2f}, {smoothed_position[2]:.2f})"
                cv2.putText(
                    debug_image,
                    pos_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )
            
            return True, hand_pose_msg, gesture_msg, debug_image
        
        else:
            # No hand detected
            hand_pose_msg = HandPose()
            hand_pose_msg.header = Header()
            hand_pose_msg.header.stamp = rospy.Time.now()
            hand_pose_msg.header.frame_id = "camera_frame"
            hand_pose_msg.hand_detected = False
            hand_pose_msg.confidence = 0.0
            
            # Draw "No hand detected" on debug image
            if debug_image is not None:
                cv2.putText(
                    debug_image,
                    "No hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
            
            return False, hand_pose_msg, None, debug_image
    
    def extract_landmarks(self, hand_landmarks, image_shape):
        """
        Extract hand landmarks as numpy array
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            image_shape: Shape of the image (height, width, channels)
            
        Returns:
            Nx3 numpy array of landmarks (normalized coordinates)
        """
        h, w, _ = image_shape
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            # MediaPipe gives normalized coordinates [0, 1]
            # We keep them normalized for now
            landmarks.append([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks)
    
    def run(self):
        """
        Main loop
        """
        rate = rospy.Rate(30)  # 30 Hz
        
        rospy.loginfo("Starting hand tracking loop...")
        
        while not rospy.is_shutdown():
            # Read frame
            ret, frame = self.cap.read()
            
            if not ret:
                rospy.logwarn("Failed to read frame from camera")
                continue
            
            # Process frame
            hand_detected, hand_pose_msg, gesture_msg, debug_image = self.process_frame(frame)
            
            # Publish hand pose
            self.hand_pose_pub.publish(hand_pose_msg)
            
            # Publish gesture if detected
            if gesture_msg is not None:
                self.gesture_pub.publish(gesture_msg)
                rospy.loginfo(f"Gesture detected: {gesture_msg.gesture_type} ({gesture_msg.confidence:.2f})")
            
            # Publish debug image
            if self.publish_debug_image and debug_image is not None:
                try:
                    debug_image_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
                    self.debug_image_pub.publish(debug_image_msg)
                except Exception as e:
                    rospy.logwarn(f"Failed to publish debug image: {e}")
            
            rate.sleep()
        
        # Cleanup
        self.cap.release()
        self.hands.close()
        rospy.loginfo("Hand Tracker Node shutdown")


if __name__ == '__main__':
    try:
        node = HandTrackerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
