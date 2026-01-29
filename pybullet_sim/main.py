#!/usr/bin/env python3
"""
Main Application - Virtual Robotic Arm Controller (PyBullet Version)
Windows-native version using PyBullet instead of ROS/Gazebo
"""

import sys
import os
import time
import subprocess
import numpy as np
import cv2

# --- AUTO-CORRECT PYTHON VERSION ---
# If running on Python 3.13+, try to switch to 3.12 automatically
if sys.version_info >= (3, 13):
    print("!" * 60)
    print("DETECTED PYTHON 3.13 (Incompatible with current MediaPipe)")
    print("Attempting to auto-switch to Python 3.12...")
    print("!" * 60)
    
    # Try finding Python 3.12 launcher
    try:
        # Re-run this script with py -3.12
        subprocess.check_call(["py", "-3.12", __file__] + sys.argv[1:])
        sys.exit(0)
    except Exception as e:
        print(f"\nCould not auto-switch: {e}")
        print("Please manually run using: py -3.12 pybullet_sim/main.py")
        print("Or double-click 'run_robot.bat'")
        print("\nContinuing with 3.13 (Hand tracking will NOT control robot)...")
        time.sleep(3)
# -----------------------------------

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catkin_ws.src.robotic_arm_controller.scripts.kinematics import create_default_robot
from pybullet_sim.robot_controller import PyBulletRobotController
from pybullet_sim.hand_tracker import HandTracker


class CoordinateMapper:
    """
    Maps hand coordinates to robot workspace
    """
    
    def __init__(self):
        # Camera workspace (normalized 0-1)
        self.camera_x_min = 0.2
        self.camera_x_max = 0.8
        self.camera_y_min = 0.2
        self.camera_y_max = 0.8
        self.camera_z_min = -0.1
        self.camera_z_max = 0.1
        
        # Robot workspace (meters)
        self.robot_x_min = 0.2
        self.robot_x_max = 0.6
        self.robot_y_min = -0.3
        self.robot_y_max = 0.3
        self.robot_z_min = 0.1
        self.robot_z_max = 0.5
        
        # Flipping
        self.flip_x = False
        self.flip_y = True
        self.flip_z = False
    
    def map_coordinate(self, value, in_min, in_max, out_min, out_max, flip=False):
        """Map value from input range to output range"""
        value = np.clip(value, in_min, in_max)
        normalized = (value - in_min) / (in_max - in_min)
        if flip:
            normalized = 1.0 - normalized
        return out_min + normalized * (out_max - out_min)
    
    def map_hand_to_robot(self, hand_position):
        """
        Map hand position to robot workspace (INTUITIVE MAPPING)
        
        Mapping:
        - Hand Left/Right (Cam X) -> Robot Left/Right (Robot Y)
        - Hand Up/Down    (Cam Y) -> Robot Up/Down    (Robot Z)
        - Hand Depth      (Cam Z) -> Robot Fwd/Back   (Robot X)
        """
        # 1. Hand X (Side) -> Robot Y (Side)
        robot_y = self.map_coordinate(
            hand_position[0],
            self.camera_x_min, self.camera_x_max,
            self.robot_y_min, self.robot_y_max,
            flip=True # Flip X so moving right moves robot right (camera is mirrored?)
        )
        
        # 2. Hand Y (Up/Down) -> Robot Z (Up/Down)
        # Note: Camera Y is 0 at top, 1 at bottom. Robot Z is 0 at bottom, up is positive.
        # So we need to flip this mapping.
        robot_z = self.map_coordinate(
            hand_position[1],
            self.camera_y_min, self.camera_y_max,
            self.robot_z_min, self.robot_z_max,
            flip=True 
        )
        
        # 3. Hand Z (Depth) -> Robot X (Forward/Backward)
        # Note: MediaPipe Z is negative when closer? Need to check.
        # Usually we map depth to reach.
        # Let's map Z directly for now, might need tuning.
        robot_x = self.map_coordinate(
            hand_position[2],
            self.camera_z_min, self.camera_z_max,
            self.robot_x_min, self.robot_x_max,
            flip=True # Closer (negative Z) should be further reach (positive X)?
        )
        
        return np.array([robot_x, robot_y, robot_z])


class VirtualRobotController:
    """
    Main controller integrating all components
    """
    
    def __init__(self):
        print("=" * 60)
        print("Virtual Robotic Arm Controller - PyBullet Version")
        print("=" * 60)
        
        
        # Initialize components
        print("\n[1/4] Initializing PyBullet simulation (Franka Panda)...")
        self.robot_sim = PyBulletRobotController(use_gui=True)
        self.robot_sim.spawn_object() # Create cube
        
        print("[2/4] Robot kinematics handled by PyBullet internal solver...")
        # self.kinematics removed as we use p.calculateInverseKinematics
        
        print("[3/4] Initializing hand tracker...")
        try:
            self.hand_tracker = HandTracker()
        except Exception as e:
            print(f"Warning: Hand tracker init failed: {e}")
        
        print("[4/4] Initializing coordinate mapper...")
        self.coord_mapper = CoordinateMapper()
        
        # State
        self.current_mode = "tracking"  # "tracking", "hold", "precision"
        self.last_valid_position = None
        self.gripper_state = 1 # 1=Open, 0=Closed
        
        # Smoothing
        self.smoothing_alpha = 0.3
        self.smoothed_target = None
        
        print("\n✓ Initialization complete!")
        print("\nControls:")
        print("  - Move your hand to control the robot")
        print("  - FIST: CLOSE GRIPPER (Hold object)")
        print("  - OPEN HAND: OPEN GRIPPER")
        print("  - Pinch: Precision mode")
        print("  - Peace: Reset to home")
        print("  - Press 'q' to quit")
        print("\n" + "=" * 60 + "\n")
    
    def smooth_position(self, new_position):
        """Apply exponential smoothing to position"""
        if self.smoothed_target is None:
            self.smoothed_target = new_position
        else:
            self.smoothed_target = (self.smoothing_alpha * new_position + 
                                   (1 - self.smoothing_alpha) * self.smoothed_target)
        return self.smoothed_target
    
    def run(self):
        """Main control loop"""
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Get hand pose
                hand_detected, hand_pos, landmarks, gesture, frame = self.hand_tracker.get_hand_pose()
                
                # Handle gestures
                if gesture:
                    gesture_name, confidence = gesture
                    
                    if gesture_name == "peace":
                        print("✓ Resetting to home position")
                        self.robot_sim.reset_robot()
                        self.smoothed_target = None
                        self.gripper_state = 1
                        continue
                    elif gesture_name == "fist":
                        self.current_mode = "hold"
                        self.gripper_state = 0 # Close gripper
                    elif gesture_name == "open":
                        self.current_mode = "tracking"
                        self.gripper_state = 1 # Open gripper
                        self.smoothing_alpha = 0.3
                    elif gesture_name == "pinch":
                        self.current_mode = "precision"
                        self.smoothing_alpha = 0.1
                
                # Apply gripper state
                self.robot_sim.control_gripper(self.gripper_state)

                # Process hand position (allow movement in all modes)
                if hand_detected:
                    # Map to robot workspace
                    robot_target = self.coord_mapper.map_hand_to_robot(hand_pos)
                    # Apply smoothing
                    smoothed_target = self.smooth_position(robot_target)
                    # VISUALIZATION: Show target in PyBullet
                    if not hasattr(self, 'target_marker'):
                        # Create a red sphere marker
                        import pybullet as p
                        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 0.7])
                        self.target_marker = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=smoothed_target)
                    else:
                        # Update marker position
                        import pybullet as p
                        p.resetBasePositionAndOrientation(self.target_marker, smoothed_target, [0, 0, 0, 1])
                    # Solve IK using PyBullet's built-in solver
                    joint_angles = self.robot_sim.calculate_ik(smoothed_target)
                    if joint_angles is not None:
                        # Send to robot
                        self.robot_sim.set_joint_positions(joint_angles)
                        self.last_valid_position = smoothed_target
                
                # Step simulation
                self.robot_sim.step_simulation()

                # --- DRAW UI OVERLAY (Always draw this) ---
                # Clean UI: Just Mode and Gripper Status
                
                # Mode
                cv2.putText(frame, f"MODE: {self.current_mode.upper()}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Gripper Status
                if self.gripper_state == 0:
                    grip_text = "GRIPPER: CLOSED"
                    grip_color = (0, 0, 255) # Red
                else:
                    grip_text = "GRIPPER: OPEN"
                    grip_color = (0, 255, 0) # Green
                    
                cv2.putText(frame, grip_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, grip_color, 2)
                # ------------------------------------------

                # Show camera feed with improved UI
                cv2.imshow('Robot Control', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                

                
                time.sleep(1./60.)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\n\nCleaning up...")
        self.hand_tracker.release()
        self.robot_sim.disconnect()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete. Goodbye!")


if __name__ == "__main__":
    try:
        controller = VirtualRobotController()
        controller.run()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
