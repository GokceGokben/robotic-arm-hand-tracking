#!/usr/bin/env python3
"""
PyBullet Robot Controller for Windows
Controls 6-DOF robot arm in PyBullet simulation
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
from pathlib import Path


class PyBulletRobotController:
    """
    PyBullet-based robot controller
    """
    
    def __init__(self, urdf_path=None, use_gui=True):
        """
        Initialize PyBullet simulation with Franka Panda
        """
        # Connect to PyBullet
        if use_gui:
            self.physics_client = p.connect(p.GUI)
            # --- CLEAN UP GUI ---
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            # Adjust camera to see the robot better
            p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.5])
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set up simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -15.0)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load Franka Emika Panda
        print("Loading Franka Emika Panda model...")
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        
        # Joint Indices for Panda
        # Arm: 0-6 (7 joints)
        # Gripper: 9, 10
        self.arm_indices = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_indices = [9, 10]
        # End effector (grasp target) is usually index 11 in this URDF
        # End effector (grasp target) is usually index 11 in this URDF
        self.ee_index = 11 
        
        # Track grasp state
        self.grasp_constraint = None
        
        # Set initial position
        self.reset_robot()
        
    def reset_robot(self):
        """Reset robot to home position"""
        # Neutral position for Panda
        home_pos = [0, -0.2, 0, -2.0, 0, 1.8, 0.8] 
        for i, idx in enumerate(self.arm_indices):
            p.resetJointState(self.robot_id, idx, home_pos[i])
        
        # Open gripper
        self.control_gripper(1) # Open

    def control_gripper(self, state):
        """
        Control gripper state
        state: 0 (Closed) or 1 (Open)
        """
        target_pos = 0.04 * state # 0.04 is open (4cm per finger), 0 is closed
        for idx in self.gripper_indices:
            p.setJointMotorControl2(
                self.robot_id, idx, p.POSITION_CONTROL, 
                targetPosition=target_pos, force=500
            )
            # Increase friction to prevent slipping
            p.changeDynamics(self.robot_id, idx, lateralFriction=2.0, spinningFriction=1.0)
            
        # Magnetic Grasp Logic
        if state == 0: # Closing
            self.attempt_grasp()
        else: # Opening
            self.release_grasp()

    def attempt_grasp(self):
        """Try to magnetically grasp object if close enough"""
        if self.grasp_constraint is not None:
            return # Already holding
            
        if not hasattr(self, 'object_id'):
            return
            
        # Get positions
        ee_pos, _ = p.getLinkState(self.robot_id, self.ee_index)[:2]
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        
        # Calculate distance
        dist = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))
        
        # If close enough (within 20cm - RELAXED), create constraint
        if dist < 0.20:
            # TELEPORT SNAP: Move object to hand for perfect grip
            p.resetBasePositionAndOrientation(self.object_id, [ee_pos[0], ee_pos[1], ee_pos[2] - 0.05], [0,0,0,1])
            
            self.grasp_constraint = p.createConstraint(
                parentBodyUniqueId=self.robot_id,
                parentLinkIndex=self.ee_index,
                childBodyUniqueId=self.object_id,
                childLinkIndex=-1,
                jointType=p.JOINT_POINT2POINT,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0], # Center of gripper
                childFramePosition=[0, 0, 0.05] # Top of object
            )
            p.changeConstraint(self.grasp_constraint, maxForce=1000)

    def release_grasp(self):
        """Release magnetic grasp"""
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None
            
            # WAKE UP PHYSICS: Ensure object falls immediately
            if hasattr(self, 'object_id'):
                # Optional: Ensure it's not "stuck" by adding a tiny downward velocity
                p.resetBaseVelocity(self.object_id, linearVelocity=[0, 0, -0.1])

    def step_simulation(self):
        """Step the simulation and update visuals"""
        self.draw_debug_lines()
        try:
            p.stepSimulation()
            return True
        except:
            return False

    def draw_debug_lines(self):
        """Draw visual feedback for grasping"""
        if hasattr(self, 'object_id'):
            # Get positions
            ee_pos, _ = p.getLinkState(self.robot_id, self.ee_index)[:2]
            obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
            
            # Distance
            dist = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))
            
            # Color: Green if in range (< 0.2), Red if far
            color = [0, 1, 0] if dist < 0.20 else [1, 0, 0]
            
            # Draw line
            p.addUserDebugLine(ee_pos, obj_pos, lineColorRGB=color, lifeTime=0.1)

    def spawn_object(self):
        """Spawn a cube to manipulate"""
        # Remove old object if exists
        if hasattr(self, 'object_id'):
            p.removeBody(self.object_id)
            
        # Create a long rectangular bar (easier to grab)
        # Dimensions: 3cm wide, 16cm long, 3cm tall
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.015, 0.08, 0.015])
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.08, 0.015], rgbaColor=[0, 1, 0, 1])
        
        # Spawn in front of robot
        self.object_id = p.createMultiBody(
            baseMass=1.0, # Heavier object (User requested bigger gravity force)
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[0.5, 0, 0.1] 
        )
        # Increase object friction
        p.changeDynamics(self.object_id, -1, lateralFriction=2.0, spinningFriction=1.0)
        
        # Track grasp state
        self.grasp_constraint = None

    def calculate_ik(self, target_pos, target_orn=None):
        """
        Calculate Inverse Kinematics using PyBullet's built-in solver
        """
        # If no orientation specified, keep gripper pointing down/forward
        if target_orn is None:
            # Fixed orientation (pointing down/forward)
            target_orn = p.getQuaternionFromEuler([3.14, 0, 0])
            
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_index,
            target_pos,
            target_orn,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # Return only the arm joint angles (first 7)
        return joint_poses[:7]

    def set_joint_positions(self, joint_positions):
        """Set arm joint positions"""
        for i, pos in enumerate(joint_positions):
            if i < len(self.arm_indices):
                p.setJointMotorControl2(
                    self.robot_id, self.arm_indices[i],
                    p.POSITION_CONTROL, targetPosition=pos,
                    force=100
                )

    def step_simulation(self):
        """Step the simulation"""
        try:
            p.stepSimulation()
            return True
        except:
            return False

    def disconnect(self):
        p.disconnect()


if __name__ == "__main__":
    # Test the controller
    print("Testing PyBullet Robot Controller...")
    
    controller = PyBulletRobotController(use_gui=True)
    
    print("\nTesting joint control...")
    
    # Test different positions
    test_positions = [
        [0, 0, 0, 0, 0, 0],
        [0.5, 0.3, -0.2, 0, 0, 0],
        [0, 0.5, 0.5, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ]
    
    for i, pos in enumerate(test_positions):
        print(f"\nMoving to position {i+1}: {pos}")
        controller.set_joint_positions(pos)
        
        # Simulate for 2 seconds
        for _ in range(240):  # 240 steps at 240Hz = 1 second
            controller.step_simulation()
            time.sleep(1./240.)
        
        # Check end-effector pose
        ee_pos, ee_orient = controller.get_end_effector_pose()
        print(f"End-effector position: {ee_pos}")
    
    print("\nTest complete. Close the window to exit.")
    
    # Keep simulation running
    try:
        while True:
            controller.step_simulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        print("\nShutting down...")
        controller.disconnect()
