#!/usr/bin/env python3
"""
PyBullet Robot Test - No Hand Tracking
Just test the robot arm movement in PyBullet
"""

import sys
import os
import time
import numpy as np

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'catkin_ws', 'src', 'robotic_arm_controller', 'scripts'))

from pybullet_sim.robot_controller import PyBulletRobotController
from kinematics import create_default_robot


def main():
    print("=" * 60)
    print("PyBullet Robot Test - No Hand Tracking")
    print("Testing robot arm movement with predefined positions")
    print("=" * 60)
    
    # Initialize
    print("\n[1/2] Initializing PyBullet simulation...")
    robot_sim = PyBulletRobotController(use_gui=True)
    
    print("[2/2] Initializing kinematics...")
    kinematics = create_default_robot()
    
    print("\n✓ Initialization complete!")
    print("\nThe robot will move through several test positions.")
    print("Close the PyBullet window to exit.\n")
    
    # Test positions in robot workspace
    test_targets = [
        np.array([0.3, 0.0, 0.3]),   # Center
        np.array([0.4, 0.2, 0.3]),   # Right
        np.array([0.4, -0.2, 0.3]),  # Left
        np.array([0.3, 0.0, 0.4]),   # Up
        np.array([0.3, 0.0, 0.2]),   # Down
        np.array([0.3, 0.0, 0.3]),   # Back to center
    ]
    
    position_names = ["Center", "Right", "Left", "Up", "Down", "Center"]
    
    try:
        for i, (target, name) in enumerate(zip(test_targets, position_names)):
            print(f"\n[{i+1}/{len(test_targets)}] Moving to: {name}")
            print(f"  Target position: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")
            
            # Solve IK
            joint_angles = kinematics.inverse_kinematics_numerical(
                target,
                method='jacobian'
            )
            
            if joint_angles is not None:
                print(f"  ✓ IK solution found")
                print(f"  Joint angles: [{', '.join([f'{a:.2f}' for a in joint_angles])}]")
                
                # Send to robot
                robot_sim.set_joint_positions(joint_angles)
                
                # Simulate for 2 seconds
                for _ in range(480):  # 480 steps at 240Hz = 2 seconds
                    if not robot_sim.step_simulation():
                        print("\n  PyBullet window closed")
                        robot_sim.disconnect()
                        return
                    time.sleep(1./240.)
                
                # Verify end-effector position
                ee_pos, ee_orient = robot_sim.get_end_effector_pose()
                error = np.linalg.norm(np.array(ee_pos) - target)
                print(f"  End-effector position: ({ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f})")
                print(f"  Position error: {error:.4f} m")
            else:
                print(f"  ✗ IK failed - target out of reach")
        
        print("\n" + "=" * 60)
        print("Test complete! Robot will hold position.")
        print("Close the PyBullet window to exit.")
        print("=" * 60)
        
        # Keep simulation running
        while True:
            if not robot_sim.step_simulation():
                print("\nPyBullet window closed")
                break
            time.sleep(1./240.)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        print("\nShutting down...")
        robot_sim.disconnect()
        print("✓ Done!")


if __name__ == "__main__":
    main()
