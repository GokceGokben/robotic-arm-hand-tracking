#!/usr/bin/env python3
"""
Kinematics Engine for 6-DOF Robotic Arm
Implements Forward Kinematics, Analytical IK, and Numerical IK using D-H parameters
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional, List
import warnings


class RobotKinematics:
    """
    6-DOF Robot Arm Kinematics using Denavit-Hartenberg parameters
    """
    
    def __init__(self, dh_params: np.ndarray):
        """
        Initialize kinematics with D-H parameters
        
        Args:
            dh_params: Nx4 array of [theta, d, a, alpha] for each joint
                      theta: joint angle (variable for revolute joints)
                      d: link offset
                      a: link length
                      alpha: link twist
        """
        self.dh_params = dh_params
        self.n_joints = len(dh_params)
        
        # Joint limits (radians) - can be configured
        self.joint_limits = np.array([
            [-np.pi, np.pi],      # Joint 1: Base rotation
            [-np.pi/2, np.pi/2],  # Joint 2: Shoulder
            [-np.pi/2, np.pi/2],  # Joint 3: Elbow
            [-np.pi, np.pi],      # Joint 4: Wrist roll
            [-np.pi/2, np.pi/2],  # Joint 5: Wrist pitch
            [-np.pi, np.pi]       # Joint 6: Wrist yaw
        ])
        
    def dh_transform(self, theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """
        Compute transformation matrix from D-H parameters
        
        Args:
            theta: joint angle
            d: link offset
            a: link length
            alpha: link twist
            
        Returns:
            4x4 homogeneous transformation matrix
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        T = np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d   ],
            [0,   0,      0,     1   ]
        ])
        
        return T
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Compute forward kinematics using D-H parameters
        
        Args:
            joint_angles: Array of 6 joint angles in radians
            
        Returns:
            Tuple of (end_effector_transform, list_of_transforms_for_each_joint)
        """
        if len(joint_angles) != self.n_joints:
            raise ValueError(f"Expected {self.n_joints} joint angles, got {len(joint_angles)}")
        
        # Initialize with identity matrix
        T = np.eye(4)
        transforms = []
        
        # Compute cumulative transformation
        for i, angle in enumerate(joint_angles):
            theta, d, a, alpha = self.dh_params[i]
            # For revolute joints, theta is the variable
            theta = angle
            
            T_i = self.dh_transform(theta, d, a, alpha)
            T = T @ T_i
            transforms.append(T.copy())
        
        return T, transforms
    
    def get_position_orientation(self, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract position and orientation from transformation matrix
        
        Args:
            T: 4x4 homogeneous transformation matrix
            
        Returns:
            Tuple of (position [x,y,z], orientation as quaternion [x,y,z,w])
        """
        position = T[:3, 3]
        
        # Convert rotation matrix to quaternion
        R = T[:3, :3]
        quaternion = self.rotation_matrix_to_quaternion(R)
        
        return position, quaternion
    
    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion [x, y, z, w]
        """
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w])
    
    def inverse_kinematics_analytical(self, target_pos: np.ndarray, 
                                     target_orient: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Analytical (closed-form) inverse kinematics solution
        
        This is a simplified analytical solution for a 6-DOF arm.
        For a general 6-DOF arm, analytical solutions can be complex.
        This implementation uses geometric approach for position and 
        decouples wrist orientation.
        
        Args:
            target_pos: Target position [x, y, z]
            target_orient: Target orientation as quaternion [x, y, z, w] (optional)
            
        Returns:
            Array of 6 joint angles, or None if no solution exists
        """
        x, y, z = target_pos
        
        # Joint 1: Base rotation (simple atan2)
        theta1 = np.arctan2(y, x)
        
        # For joints 2-3, use geometric approach (simplified)
        # This assumes a planar 2-link arm for the shoulder-elbow
        r = np.sqrt(x**2 + y**2)
        
        # Get link lengths from D-H parameters
        a2 = self.dh_params[1, 2]  # Link 2 length
        a3 = self.dh_params[2, 2]  # Link 3 length
        
        # Distance to target in the plane
        d = np.sqrt(r**2 + z**2)
        
        # Check if target is reachable
        if d > (a2 + a3) or d < abs(a2 - a3):
            warnings.warn("Target position is out of reach")
            return None
        
        # Elbow angle using law of cosines
        cos_theta3 = (d**2 - a2**2 - a3**2) / (2 * a2 * a3)
        cos_theta3 = np.clip(cos_theta3, -1, 1)  # Numerical stability
        theta3 = np.arccos(cos_theta3)
        
        # Shoulder angle
        alpha = np.arctan2(z, r)
        beta = np.arctan2(a3 * np.sin(theta3), a2 + a3 * np.cos(theta3))
        theta2 = alpha - beta
        
        # For wrist joints (4-6), we need the target orientation
        # Simplified: set to neutral position if no orientation specified
        if target_orient is None:
            theta4 = 0
            theta5 = 0
            theta6 = 0
        else:
            # Compute wrist orientation from target orientation and current arm orientation
            # This requires computing the rotation from base to wrist center
            # Simplified implementation: use Euler angles
            theta4 = 0  # Placeholder
            theta5 = 0  # Placeholder
            theta6 = 0  # Placeholder
        
        joint_angles = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
        
        # Check joint limits
        if not self.check_joint_limits(joint_angles):
            warnings.warn("Solution violates joint limits")
            return None
        
        return joint_angles
    
    def inverse_kinematics_numerical(self, target_pos: np.ndarray, 
                                    target_orient: Optional[np.ndarray] = None,
                                    initial_guess: Optional[np.ndarray] = None,
                                    method: str = 'jacobian') -> Optional[np.ndarray]:
        """
        Numerical inverse kinematics using optimization or Jacobian method
        
        Args:
            target_pos: Target position [x, y, z]
            target_orient: Target orientation as quaternion [x, y, z, w] (optional)
            initial_guess: Initial joint angles (if None, uses zeros)
            method: 'jacobian' for Jacobian-based or 'optimize' for optimization
            
        Returns:
            Array of 6 joint angles, or None if no solution found
        """
        if initial_guess is None:
            initial_guess = np.zeros(self.n_joints)
        
        if method == 'jacobian':
            return self._ik_jacobian(target_pos, target_orient, initial_guess)
        else:
            return self._ik_optimization(target_pos, target_orient, initial_guess)
    
    def _ik_jacobian(self, target_pos: np.ndarray, target_orient: Optional[np.ndarray],
                    initial_guess: np.ndarray, max_iterations: int = 100,
                    tolerance: float = 1e-4) -> Optional[np.ndarray]:
        """
        Jacobian-based IK using damped least squares (Levenberg-Marquardt)
        """
        q = initial_guess.copy()
        lambda_damping = 0.01
        
        for iteration in range(max_iterations):
            # Compute current end-effector position
            T, _ = self.forward_kinematics(q)
            current_pos, current_orient = self.get_position_orientation(T)
            
            # Position error
            pos_error = target_pos - current_pos
            
            # Check convergence
            if np.linalg.norm(pos_error) < tolerance:
                return q
            
            # Compute Jacobian
            J = self.compute_jacobian(q)
            
            # Use only position part of Jacobian (first 3 rows)
            J_pos = J[:3, :]
            
            # Damped least squares solution
            delta_q = J_pos.T @ np.linalg.inv(
                J_pos @ J_pos.T + lambda_damping**2 * np.eye(3)
            ) @ pos_error
            
            # Update joint angles
            q = q + delta_q
            
            # Enforce joint limits
            q = self.enforce_joint_limits(q)
        
        warnings.warn(f"IK did not converge after {max_iterations} iterations")
        return q  # Return best effort
    
    def _ik_optimization(self, target_pos: np.ndarray, target_orient: Optional[np.ndarray],
                        initial_guess: np.ndarray) -> Optional[np.ndarray]:
        """
        Optimization-based IK using scipy.optimize
        """
        def objective(q):
            T, _ = self.forward_kinematics(q)
            current_pos, _ = self.get_position_orientation(T)
            return np.linalg.norm(target_pos - current_pos)**2
        
        # Bounds from joint limits
        bounds = [(self.joint_limits[i, 0], self.joint_limits[i, 1]) 
                 for i in range(self.n_joints)]
        
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds)
        
        if result.success and result.fun < 1e-6:
            return result.x
        else:
            warnings.warn("Optimization-based IK did not find a good solution")
            return result.x  # Return best effort
    
    def compute_jacobian(self, joint_angles: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """
        Compute Jacobian matrix using numerical differentiation
        
        Args:
            joint_angles: Current joint angles
            epsilon: Small perturbation for numerical derivative
            
        Returns:
            6x6 Jacobian matrix (3 for position, 3 for orientation)
        """
        J = np.zeros((6, self.n_joints))
        
        # Get current end-effector pose
        T0, _ = self.forward_kinematics(joint_angles)
        pos0, _ = self.get_position_orientation(T0)
        
        # Numerical differentiation for each joint
        for i in range(self.n_joints):
            q_perturbed = joint_angles.copy()
            q_perturbed[i] += epsilon
            
            T_perturbed, _ = self.forward_kinematics(q_perturbed)
            pos_perturbed, _ = self.get_position_orientation(T_perturbed)
            
            # Position derivative
            J[:3, i] = (pos_perturbed - pos0) / epsilon
            
            # Orientation derivative (simplified - angular velocity)
            # For full implementation, compute rotation difference
            J[3:, i] = 0  # Placeholder
        
        return J
    
    def check_joint_limits(self, joint_angles: np.ndarray) -> bool:
        """Check if joint angles are within limits"""
        for i, angle in enumerate(joint_angles):
            if angle < self.joint_limits[i, 0] or angle > self.joint_limits[i, 1]:
                return False
        return True
    
    def enforce_joint_limits(self, joint_angles: np.ndarray) -> np.ndarray:
        """Clamp joint angles to limits"""
        clamped = joint_angles.copy()
        for i in range(self.n_joints):
            clamped[i] = np.clip(clamped[i], 
                               self.joint_limits[i, 0], 
                               self.joint_limits[i, 1])
        return clamped


def create_default_robot() -> RobotKinematics:
    """
    Create a default 6-DOF robot with standard D-H parameters
    
    This is a sample configuration - adjust based on your actual robot
    """
    # D-H parameters: [theta, d, a, alpha]
    # theta: joint angle (variable)
    # d: link offset along z
    # a: link length along x
    # alpha: link twist about x
    
    dh_params = np.array([
        [0,     0.1,  0,     np.pi/2],   # Joint 1: Base
        [0,     0,    0.3,   0],          # Joint 2: Shoulder
        [0,     0,    0.25,  0],          # Joint 3: Elbow
        [0,     0.15, 0,     np.pi/2],   # Joint 4: Wrist roll
        [0,     0,    0,     -np.pi/2],  # Joint 5: Wrist pitch
        [0,     0.1,  0,     0]           # Joint 6: Wrist yaw
    ])
    
    return RobotKinematics(dh_params)


if __name__ == "__main__":
    # Test the kinematics
    robot = create_default_robot()
    
    # Test forward kinematics
    print("=== Testing Forward Kinematics ===")
    test_angles = np.array([0, 0, 0, 0, 0, 0])
    T, _ = robot.forward_kinematics(test_angles)
    pos, orient = robot.get_position_orientation(T)
    print(f"Joint angles: {test_angles}")
    print(f"End-effector position: {pos}")
    print(f"End-effector orientation (quaternion): {orient}")
    
    # Test analytical IK
    print("\n=== Testing Analytical IK ===")
    target_pos = np.array([0.3, 0.2, 0.3])
    solution_analytical = robot.inverse_kinematics_analytical(target_pos)
    if solution_analytical is not None:
        print(f"Target position: {target_pos}")
        print(f"IK solution: {solution_analytical}")
        
        # Verify solution
        T_verify, _ = robot.forward_kinematics(solution_analytical)
        pos_verify, _ = robot.get_position_orientation(T_verify)
        print(f"Achieved position: {pos_verify}")
        print(f"Position error: {np.linalg.norm(target_pos - pos_verify)}")
    
    # Test numerical IK
    print("\n=== Testing Numerical IK (Jacobian) ===")
    solution_numerical = robot.inverse_kinematics_numerical(target_pos, method='jacobian')
    if solution_numerical is not None:
        print(f"IK solution: {solution_numerical}")
        
        # Verify solution
        T_verify, _ = robot.forward_kinematics(solution_numerical)
        pos_verify, _ = robot.get_position_orientation(T_verify)
        print(f"Achieved position: {pos_verify}")
        print(f"Position error: {np.linalg.norm(target_pos - pos_verify)}")
