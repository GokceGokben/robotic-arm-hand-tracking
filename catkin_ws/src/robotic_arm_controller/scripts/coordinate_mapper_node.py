#!/usr/bin/env python3
"""
Coordinate Mapper ROS Node
Maps hand coordinates from camera space to robot workspace
"""

import rospy
import numpy as np
from robotic_arm_controller.msg import HandPose
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import Header
from dynamic_reconfigure.server import Server
# from robotic_arm_controller.cfg import CoordinateMapperConfig  # Will be created later

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from transform_utils import ExponentialFilter


class CoordinateMapperNode:
    """
    ROS node for mapping hand coordinates to robot workspace
    """
    
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('coordinate_mapper_node', anonymous=True)
        
        # Parameters - Camera workspace bounds (normalized coordinates)
        self.camera_x_min = rospy.get_param('~camera_x_min', 0.2)
        self.camera_x_max = rospy.get_param('~camera_x_max', 0.8)
        self.camera_y_min = rospy.get_param('~camera_y_min', 0.2)
        self.camera_y_max = rospy.get_param('~camera_y_max', 0.8)
        self.camera_z_min = rospy.get_param('~camera_z_min', -0.1)
        self.camera_z_max = rospy.get_param('~camera_z_max', 0.1)
        
        # Robot workspace bounds (meters)
        self.robot_x_min = rospy.get_param('~robot_x_min', 0.2)
        self.robot_x_max = rospy.get_param('~robot_x_max', 0.6)
        self.robot_y_min = rospy.get_param('~robot_y_min', -0.3)
        self.robot_y_max = rospy.get_param('~robot_y_max', 0.3)
        self.robot_z_min = rospy.get_param('~robot_z_min', 0.1)
        self.robot_z_max = rospy.get_param('~robot_z_max', 0.5)
        
        # Coordinate transformation parameters
        self.flip_x = rospy.get_param('~flip_x', False)
        self.flip_y = rospy.get_param('~flip_y', True)  # Usually need to flip Y
        self.flip_z = rospy.get_param('~flip_z', False)
        
        # Smoothing
        self.smoothing_alpha = rospy.get_param('~smoothing_alpha', 0.2)
        self.position_filter = ExponentialFilter(alpha=self.smoothing_alpha)
        
        # Publishers
        self.target_pose_pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=10)
        
        # Subscribers
        self.hand_pose_sub = rospy.Subscriber('/hand_pose', HandPose, self.hand_pose_callback)
        
        # Dynamic reconfigure (optional - for runtime calibration)
        # self.dyn_reconf_srv = Server(CoordinateMapperConfig, self.dynamic_reconfigure_callback)
        
        rospy.loginfo("Coordinate Mapper Node initialized")
        rospy.loginfo(f"Camera workspace: X[{self.camera_x_min}, {self.camera_x_max}], "
                     f"Y[{self.camera_y_min}, {self.camera_y_max}], "
                     f"Z[{self.camera_z_min}, {self.camera_z_max}]")
        rospy.loginfo(f"Robot workspace: X[{self.robot_x_min}, {self.robot_x_max}], "
                     f"Y[{self.robot_y_min}, {self.robot_y_max}], "
                     f"Z[{self.robot_z_min}, {self.robot_z_max}]")
    
    def map_coordinate(self, value, in_min, in_max, out_min, out_max, flip=False):
        """
        Map a value from input range to output range
        
        Args:
            value: Input value
            in_min, in_max: Input range
            out_min, out_max: Output range
            flip: If True, flip the mapping
            
        Returns:
            Mapped value
        """
        # Clamp to input range
        value = np.clip(value, in_min, in_max)
        
        # Normalize to [0, 1]
        normalized = (value - in_min) / (in_max - in_min)
        
        # Flip if requested
        if flip:
            normalized = 1.0 - normalized
        
        # Map to output range
        mapped = out_min + normalized * (out_max - out_min)
        
        return mapped
    
    def hand_pose_callback(self, msg):
        """
        Callback for hand pose messages
        
        Args:
            msg: HandPose message
        """
        if not msg.hand_detected:
            # Don't publish if no hand detected
            return
        
        # Extract hand position (normalized camera coordinates)
        hand_x = msg.position.x
        hand_y = msg.position.y
        hand_z = msg.position.z
        
        # Map to robot workspace
        robot_x = self.map_coordinate(
            hand_x,
            self.camera_x_min, self.camera_x_max,
            self.robot_x_min, self.robot_x_max,
            flip=self.flip_x
        )
        
        robot_y = self.map_coordinate(
            hand_y,
            self.camera_y_min, self.camera_y_max,
            self.robot_y_min, self.robot_y_max,
            flip=self.flip_y
        )
        
        robot_z = self.map_coordinate(
            hand_z,
            self.camera_z_min, self.camera_z_max,
            self.robot_z_min, self.robot_z_max,
            flip=self.flip_z
        )
        
        # Create position vector
        robot_position = np.array([robot_x, robot_y, robot_z])
        
        # Apply smoothing
        smoothed_position = self.position_filter.update(robot_position)
        
        # Create target pose message
        target_pose = PoseStamped()
        target_pose.header = Header()
        target_pose.header.stamp = rospy.Time.now()
        target_pose.header.frame_id = "base_link"
        
        target_pose.pose.position = Point(
            x=smoothed_position[0],
            y=smoothed_position[1],
            z=smoothed_position[2]
        )
        
        # For now, use fixed orientation (pointing down)
        # Can be improved to use hand orientation
        target_pose.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        
        # Publish
        self.target_pose_pub.publish(target_pose)
        
        # Log occasionally
        if rospy.get_time() % 2 < 0.1:  # Log every ~2 seconds
            rospy.loginfo(f"Hand: ({hand_x:.2f}, {hand_y:.2f}, {hand_z:.2f}) -> "
                         f"Robot: ({smoothed_position[0]:.3f}, {smoothed_position[1]:.3f}, {smoothed_position[2]:.3f})")
    
    def dynamic_reconfigure_callback(self, config, level):
        """
        Callback for dynamic reconfigure
        Allows runtime adjustment of mapping parameters
        """
        rospy.loginfo("Reconfigure request")
        
        # Update parameters
        self.camera_x_min = config.camera_x_min
        self.camera_x_max = config.camera_x_max
        self.camera_y_min = config.camera_y_min
        self.camera_y_max = config.camera_y_max
        self.camera_z_min = config.camera_z_min
        self.camera_z_max = config.camera_z_max
        
        self.robot_x_min = config.robot_x_min
        self.robot_x_max = config.robot_x_max
        self.robot_y_min = config.robot_y_min
        self.robot_y_max = config.robot_y_max
        self.robot_z_min = config.robot_z_min
        self.robot_z_max = config.robot_z_max
        
        self.flip_x = config.flip_x
        self.flip_y = config.flip_y
        self.flip_z = config.flip_z
        
        return config
    
    def run(self):
        """
        Main loop
        """
        rospy.loginfo("Coordinate Mapper running...")
        rospy.spin()


if __name__ == '__main__':
    try:
        node = CoordinateMapperNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
