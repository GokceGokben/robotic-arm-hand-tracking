#!/usr/bin/env python3
"""
Transformation Utilities for Robot Kinematics
Provides functions for rotation conversions and coordinate transformations
"""

import numpy as np
from typing import Tuple


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to rotation matrix
    Uses ZYX convention (yaw-pitch-roll)
    
    Args:
        roll: Rotation around X axis (radians)
        pitch: Rotation around Y axis (radians)
        yaw: Rotation around Z axis (radians)
        
    Returns:
        3x3 rotation matrix
    """
    # Rotation around X (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Rotation around Y (pitch)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Rotation around Z (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles (roll, pitch, yaw)
    Uses ZYX convention
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Check for gimbal lock
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return roll, pitch, yaw


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix
    
    Args:
        q: Quaternion [x, y, z, w]
        
    Returns:
        3x3 rotation matrix
    """
    x, y, z, w = q
    
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion [x, y, z, w]
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion [x, y, z, w]
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


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles to quaternion
    
    Args:
        roll, pitch, yaw: Euler angles in radians
        
    Returns:
        Quaternion [x, y, z, w]
    """
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    return rotation_matrix_to_quaternion(R)


def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles
    
    Args:
        q: Quaternion [x, y, z, w]
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    R = quaternion_to_rotation_matrix(q)
    return rotation_matrix_to_euler(R)


def create_transform_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """
    Create 4x4 homogeneous transformation matrix
    
    Args:
        position: [x, y, z]
        rotation: 3x3 rotation matrix
        
    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return T


def decompose_transform_matrix(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose transformation matrix into position and rotation
    
    Args:
        T: 4x4 transformation matrix
        
    Returns:
        Tuple of (position, rotation_matrix)
    """
    position = T[:3, 3]
    rotation = T[:3, :3]
    return position, rotation


class MovingAverageFilter:
    """Simple moving average filter for smoothing data"""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buffer = []
    
    def update(self, value: np.ndarray) -> np.ndarray:
        """Add new value and return filtered result"""
        self.buffer.append(value.copy())
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        return np.mean(self.buffer, axis=0)
    
    def reset(self):
        """Clear the buffer"""
        self.buffer = []


class ExponentialFilter:
    """Exponential moving average filter"""
    
    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: Smoothing factor (0-1). Lower = smoother but more lag
        """
        self.alpha = alpha
        self.filtered_value = None
    
    def update(self, value: np.ndarray) -> np.ndarray:
        """Add new value and return filtered result"""
        if self.filtered_value is None:
            self.filtered_value = value.copy()
        else:
            self.filtered_value = self.alpha * value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value
    
    def reset(self):
        """Reset the filter"""
        self.filtered_value = None


if __name__ == "__main__":
    # Test transformations
    print("=== Testing Transformation Utilities ===")
    
    # Test Euler <-> Rotation Matrix
    roll, pitch, yaw = 0.1, 0.2, 0.3
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    roll2, pitch2, yaw2 = rotation_matrix_to_euler(R)
    print(f"Euler angles: ({roll:.3f}, {pitch:.3f}, {yaw:.3f})")
    print(f"Recovered: ({roll2:.3f}, {pitch2:.3f}, {yaw2:.3f})")
    
    # Test Quaternion <-> Rotation Matrix
    q = euler_to_quaternion(roll, pitch, yaw)
    R2 = quaternion_to_rotation_matrix(q)
    print(f"\nQuaternion: {q}")
    print(f"Rotation matrices match: {np.allclose(R, R2)}")
    
    # Test filters
    print("\n=== Testing Filters ===")
    ma_filter = MovingAverageFilter(window_size=3)
    exp_filter = ExponentialFilter(alpha=0.5)
    
    test_data = [np.array([1.0, 2.0, 3.0]),
                 np.array([1.5, 2.5, 3.5]),
                 np.array([1.2, 2.2, 3.2])]
    
    for data in test_data:
        ma_result = ma_filter.update(data)
        exp_result = exp_filter.update(data)
        print(f"Input: {data}, MA: {ma_result}, Exp: {exp_result}")
