#!/usr/bin/env python3
"""
Robot Controller ROS Node
Controls the robot arm in Gazebo simulation
Receives joint commands and publishes to Gazebo controllers
"""

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from transform_utils import ExponentialFilter


class RobotControllerNode:
    """
    ROS node for robot control
    """
    
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('robot_controller_node', anonymous=True)
        
        # Parameters
        self.control_mode = rospy.get_param('~control_mode', 'position')  # 'position' or 'trajectory'
        self.smoothing_alpha = rospy.get_param('~smoothing_alpha', 0.3)
        self.max_joint_velocity = rospy.get_param('~max_joint_velocity', 1.0)  # rad/s
        
        # Joint names
        self.joint_names = [
            'joint_1', 'joint_2', 'joint_3',
            'joint_4', 'joint_5', 'joint_6'
        ]
        
        # Current and target joint positions
        self.current_joint_positions = np.zeros(6)
        self.target_joint_positions = np.zeros(6)
        
        # Smoothing filters for each joint
        self.joint_filters = [ExponentialFilter(alpha=self.smoothing_alpha) for _ in range(6)]
        
        # Publishers - for position control mode
        self.joint_pubs = []
        for joint_name in self.joint_names:
            pub = rospy.Publisher(f'/{joint_name}_position_controller/command', Float64, queue_size=10)
            self.joint_pubs.append(pub)
        
        # Publisher - for trajectory control mode
        self.trajectory_pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)
        
        # Subscribers
        self.joint_commands_sub = rospy.Subscriber('/joint_commands', JointState, self.joint_commands_callback)
        self.joint_states_sub = rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        
        rospy.loginfo("Robot Controller Node initialized")
        rospy.loginfo(f"Control mode: {self.control_mode}")
        rospy.loginfo(f"Max joint velocity: {self.max_joint_velocity} rad/s")
    
    def joint_states_callback(self, msg):
        """
        Callback for joint states
        
        Args:
            msg: JointState message
        """
        # Update current joint positions
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.current_joint_positions[i] = msg.position[idx]
    
    def joint_commands_callback(self, msg):
        """
        Callback for joint commands
        
        Args:
            msg: JointState message with target joint positions
        """
        if len(msg.position) < 6:
            rospy.logwarn(f"Received joint command with {len(msg.position)} joints, expected 6")
            return
        
        # Update target positions
        self.target_joint_positions = np.array(msg.position[:6])
        
        # Apply smoothing
        smoothed_positions = []
        for i in range(6):
            smoothed = self.joint_filters[i].update(np.array([self.target_joint_positions[i]]))
            smoothed_positions.append(smoothed[0])
        
        smoothed_positions = np.array(smoothed_positions)
        
        # Publish commands based on control mode
        if self.control_mode == 'position':
            self.publish_position_commands(smoothed_positions)
        else:
            self.publish_trajectory_command(smoothed_positions)
    
    def publish_position_commands(self, positions):
        """
        Publish position commands to individual joint controllers
        
        Args:
            positions: Array of 6 joint positions
        """
        for i, pub in enumerate(self.joint_pubs):
            msg = Float64()
            msg.data = positions[i]
            pub.publish(msg)
    
    def publish_trajectory_command(self, positions):
        """
        Publish trajectory command
        
        Args:
            positions: Array of 6 joint positions
        """
        # Calculate time to reach target based on max velocity
        position_diff = np.abs(positions - self.current_joint_positions)
        max_diff = np.max(position_diff)
        time_to_target = max_diff / self.max_joint_velocity if max_diff > 0 else 0.1
        time_to_target = max(time_to_target, 0.1)  # Minimum 0.1 seconds
        
        # Create trajectory message
        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = self.joint_names
        
        # Add trajectory point
        point = JointTrajectoryPoint()
        point.positions = positions.tolist()
        point.time_from_start = rospy.Duration(time_to_target)
        
        traj.points.append(point)
        
        # Publish
        self.trajectory_pub.publish(traj)
    
    def run(self):
        """
        Main loop
        """
        rospy.loginfo("Robot Controller running...")
        rospy.spin()


if __name__ == '__main__':
    try:
        node = RobotControllerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
