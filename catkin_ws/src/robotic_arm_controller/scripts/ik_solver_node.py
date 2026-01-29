#!/usr/bin/env python3
"""
IK Solver ROS Node
Solves inverse kinematics for target poses
Supports both analytical and numerical methods
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from kinematics import RobotKinematics, create_default_robot


class IKSolverNode:
    """
    ROS node for inverse kinematics solving
    """
    
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('ik_solver_node', anonymous=True)
        
        # Parameters
        self.solver_method = rospy.get_param('~solver_method', 'numerical')  # 'analytical' or 'numerical'
        self.use_current_as_seed = rospy.get_param('~use_current_as_seed', True)
        
        # Initialize kinematics
        self.robot = create_default_robot()
        
        # Current joint state (used as seed for numerical IK)
        self.current_joint_state = np.zeros(6)
        self.joint_state_received = False
        
        # Publishers
        self.joint_commands_pub = rospy.Publisher('/joint_commands', JointState, queue_size=10)
        
        # Subscribers
        self.target_pose_sub = rospy.Subscriber('/target_pose', PoseStamped, self.target_pose_callback)
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        
        # Statistics
        self.solve_count = 0
        self.success_count = 0
        self.total_solve_time = 0.0
        
        rospy.loginfo("IK Solver Node initialized")
        rospy.loginfo(f"Solver method: {self.solver_method}")
        rospy.loginfo(f"Use current joint state as seed: {self.use_current_as_seed}")
    
    def joint_state_callback(self, msg):
        """
        Callback for joint state messages
        
        Args:
            msg: JointState message
        """
        if len(msg.position) >= 6:
            self.current_joint_state = np.array(msg.position[:6])
            self.joint_state_received = True
    
    def target_pose_callback(self, msg):
        """
        Callback for target pose messages
        Solves IK and publishes joint commands
        
        Args:
            msg: PoseStamped message
        """
        # Extract target position
        target_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        # Extract target orientation (quaternion)
        target_orient = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        
        # Solve IK
        start_time = rospy.get_time()
        
        if self.solver_method == 'analytical':
            solution = self.robot.inverse_kinematics_analytical(target_pos, target_orient)
        else:  # numerical
            # Use current joint state as initial guess if available
            initial_guess = self.current_joint_state if (self.use_current_as_seed and self.joint_state_received) else None
            solution = self.robot.inverse_kinematics_numerical(target_pos, target_orient, initial_guess, method='jacobian')
        
        solve_time = rospy.get_time() - start_time
        
        # Update statistics
        self.solve_count += 1
        self.total_solve_time += solve_time
        
        if solution is not None:
            self.success_count += 1
            
            # Create joint command message
            joint_cmd = JointState()
            joint_cmd.header = Header()
            joint_cmd.header.stamp = rospy.Time.now()
            
            joint_cmd.name = [
                'joint_1', 'joint_2', 'joint_3',
                'joint_4', 'joint_5', 'joint_6'
            ]
            joint_cmd.position = solution.tolist()
            
            # Publish
            self.joint_commands_pub.publish(joint_cmd)
            
            # Log occasionally
            if self.solve_count % 30 == 0:  # Every 30 solves
                avg_time = self.total_solve_time / self.solve_count
                success_rate = (self.success_count / self.solve_count) * 100
                rospy.loginfo(f"IK Stats: {self.solve_count} solves, "
                             f"{success_rate:.1f}% success, "
                             f"{avg_time*1000:.2f}ms avg solve time")
        else:
            rospy.logwarn(f"IK failed for target position: {target_pos}")
    
    def run(self):
        """
        Main loop
        """
        rospy.loginfo("IK Solver running...")
        rospy.spin()


if __name__ == '__main__':
    try:
        node = IKSolverNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
