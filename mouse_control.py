#!/usr/bin/env python3
"""
Mouse-Controlled Robot Arm - Works with Python 3.13
Control the robot arm with your mouse instead of hand tracking
"""

import sys
import os
import time
import numpy as np
import cv2

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'catkin_ws', 'src', 'robotic_arm_controller', 'scripts'))

from pybullet_sim.robot_controller import PyBulletRobotController
from kinematics import create_default_robot


class MouseController:
    """Control robot with mouse"""
    
    def __init__(self):
        self.mouse_x = 0.5
        self.mouse_y = 0.5
        self.mouse_z = 0.5
        self.window_name = "Mouse Control"
        
        # Create control window
        self.control_image = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Workspace bounds
        self.robot_x_min = 0.2
        self.robot_x_max = 0.6
        self.robot_y_min = -0.3
        self.robot_y_max = 0.3
        self.robot_z_min = 0.1
        self.robot_z_max = 0.5
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_MOUSEMOVE:
            # Update X and Y from mouse position
            self.mouse_x = x / 800.0
            self.mouse_y = y / 600.0
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Update Z from mouse wheel
            if flags > 0:
                self.mouse_z = min(1.0, self.mouse_z + 0.05)
            else:
                self.mouse_z = max(0.0, self.mouse_z - 0.05)
    
    def get_target_position(self):
        """Convert mouse position to robot workspace"""
        robot_x = self.robot_x_min + self.mouse_x * (self.robot_x_max - self.robot_x_min)
        robot_y = self.robot_y_min + (1.0 - self.mouse_y) * (self.robot_y_max - self.robot_y_min)
        robot_z = self.robot_z_min + self.mouse_z * (self.robot_z_max - self.robot_z_min)
        
        return np.array([robot_x, robot_y, robot_z])
    
    def draw_control_panel(self, target_pos, ik_success):
        """Draw the control panel"""
        self.control_image.fill(40)
        
        # Title
        cv2.putText(self.control_image, "Mouse-Controlled Robot Arm", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Instructions
        y_pos = 100
        instructions = [
            "Controls:",
            "  - Move mouse to control X and Y position",
            "  - Scroll wheel to control Z (height)",
            "  - Press 'r' to reset to center",
            "  - Press 'q' to quit",
            "",
            "Current Target Position:",
            f"  X: {target_pos[0]:.3f} m",
            f"  Y: {target_pos[1]:.3f} m", 
            f"  Z: {target_pos[2]:.3f} m",
        ]
        
        for instruction in instructions:
            cv2.putText(self.control_image, instruction, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_pos += 35
        
        # IK Status
        y_pos += 20
        if ik_success:
            status_text = "IK: SUCCESS"
            status_color = (0, 255, 0)
        else:
            status_text = "IK: OUT OF REACH"
            status_color = (0, 0, 255)
        
        cv2.putText(self.control_image, status_text, (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        
        # Visualization
        # Draw workspace bounds
        workspace_x = 50
        workspace_y = 400
        workspace_w = 700
        workspace_h = 150
        
        cv2.rectangle(self.control_image, 
                     (workspace_x, workspace_y),
                     (workspace_x + workspace_w, workspace_y + workspace_h),
                     (100, 100, 100), 2)
        
        cv2.putText(self.control_image, "Robot Workspace (Top View)", 
                   (workspace_x, workspace_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        # Draw target position
        target_x = int(workspace_x + (target_pos[0] - self.robot_x_min) / 
                      (self.robot_x_max - self.robot_x_min) * workspace_w)
        target_y = int(workspace_y + (1.0 - (target_pos[1] - self.robot_y_min) / 
                      (self.robot_y_max - self.robot_y_min)) * workspace_h)
        
        cv2.circle(self.control_image, (target_x, target_y), 10, (0, 255, 255), -1)
        cv2.circle(self.control_image, (target_x, target_y), 12, (255, 255, 0), 2)
        
        # Height indicator
        height_x = 750
        height_y = 400
        height_h = 150
        
        cv2.rectangle(self.control_image,
                     (height_x - 20, height_y),
                     (height_x + 20, height_y + height_h),
                     (100, 100, 100), 2)
        
        cv2.putText(self.control_image, "Z", (height_x - 10, height_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        z_indicator = int(height_y + (1.0 - (target_pos[2] - self.robot_z_min) / 
                         (self.robot_z_max - self.robot_z_min)) * height_h)
        
        cv2.rectangle(self.control_image,
                     (height_x - 20, z_indicator - 5),
                     (height_x + 20, z_indicator + 5),
                     (0, 255, 255), -1)
        
        return self.control_image


def main():
    print("=" * 60)
    print("Mouse-Controlled Robot Arm - Python 3.13 Compatible")
    print("=" * 60)
    
    # Initialize
    print("\n[1/3] Initializing PyBullet simulation...")
    robot_sim = PyBulletRobotController(use_gui=True)
    
    print("[2/3] Initializing kinematics...")
    kinematics = create_default_robot()
    
    print("[3/3] Initializing mouse controller...")
    mouse_ctrl = MouseController()
    
    print("\n✓ Initialization complete!")
    print("\nMove your mouse in the control window to move the robot!")
    print("Scroll wheel to adjust height (Z axis)")
    print("Press 'q' to quit\n")
    
    # Smoothing
    smoothed_target = None
    smoothing_alpha = 0.3
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Get target from mouse
            target_pos = mouse_ctrl.get_target_position()
            
            # Apply smoothing
            if smoothed_target is None:
                smoothed_target = target_pos
            else:
                smoothed_target = (smoothing_alpha * target_pos + 
                                 (1 - smoothing_alpha) * smoothed_target)
            
            # Solve IK
            joint_angles = kinematics.inverse_kinematics_numerical(
                smoothed_target,
                method='jacobian'
            )
            
            ik_success = joint_angles is not None
            
            if ik_success:
                # Send to robot
                robot_sim.set_joint_positions(joint_angles)
            
            # Step simulation
            if not robot_sim.step_simulation():
                print("\nPyBullet window closed")
                break
            
            # Draw control panel
            control_image = mouse_ctrl.draw_control_panel(smoothed_target, ik_success)
            cv2.imshow(mouse_ctrl.window_name, control_image)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset to center
                mouse_ctrl.mouse_x = 0.5
                mouse_ctrl.mouse_y = 0.5
                mouse_ctrl.mouse_z = 0.5
                smoothed_target = None
            
            # FPS counter
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"FPS: {fps:.1f} | Target: ({smoothed_target[0]:.2f}, {smoothed_target[1]:.2f}, {smoothed_target[2]:.2f}) | IK: {'OK' if ik_success else 'FAIL'}", end='\r')
            
            time.sleep(1./60.)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        print("\n\nCleaning up...")
        robot_sim.disconnect()
        cv2.destroyAllWindows()
        print("✓ Done!")


if __name__ == "__main__":
    main()
