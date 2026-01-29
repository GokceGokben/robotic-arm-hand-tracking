#!/usr/bin/env python3
"""
Hardware Bridge ROS Node
Bridges ROS to Arduino/Raspberry Pi for physical robot control
Uses rosserial for communication
"""

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Bool
import serial
import struct
import numpy as np


class HardwareBridgeNode:
    """
    ROS node for hardware communication
    """
    
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('hardware_bridge_node', anonymous=True)
        
        # Parameters
        self.serial_port = rospy.get_param('~serial_port', '/dev/ttyUSB0')
        self.baud_rate = rospy.get_param('~baud_rate', 115200)
        self.enable_hardware = rospy.get_param('~enable_hardware', True)
        
        # Joint angle limits (degrees) for servos
        self.servo_min = rospy.get_param('~servo_min', [0, 0, 0, 0, 0, 0])
        self.servo_max = rospy.get_param('~servo_max', [180, 180, 180, 180, 180, 180])
        
        # Mapping from radians to servo angles
        self.joint_limits_rad = np.array([
            [-np.pi, np.pi],
            [-np.pi/2, np.pi/2],
            [-np.pi/2, np.pi/2],
            [-np.pi, np.pi],
            [-np.pi/2, np.pi/2],
            [-np.pi, np.pi]
        ])
        
        # Serial connection
        self.ser = None
        if self.enable_hardware:
            try:
                self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
                rospy.loginfo(f"Connected to hardware on {self.serial_port}")
            except Exception as e:
                rospy.logerr(f"Failed to connect to hardware: {e}")
                self.enable_hardware = False
        
        # Publishers
        self.hardware_status_pub = rospy.Publisher('/hardware_status', Bool, queue_size=1)
        
        # Subscribers
        self.joint_commands_sub = rospy.Subscriber('/joint_commands', JointState, self.joint_commands_callback)
        
        # Emergency stop flag
        self.emergency_stop = False
        
        rospy.loginfo("Hardware Bridge Node initialized")
        rospy.loginfo(f"Hardware enabled: {self.enable_hardware}")
    
    def rad_to_servo_angle(self, joint_idx, angle_rad):
        """
        Convert joint angle in radians to servo angle in degrees
        
        Args:
            joint_idx: Joint index (0-5)
            angle_rad: Angle in radians
            
        Returns:
            Servo angle in degrees
        """
        # Clamp to joint limits
        angle_rad = np.clip(angle_rad, 
                           self.joint_limits_rad[joint_idx, 0],
                           self.joint_limits_rad[joint_idx, 1])
        
        # Normalize to [0, 1]
        normalized = (angle_rad - self.joint_limits_rad[joint_idx, 0]) / \
                    (self.joint_limits_rad[joint_idx, 1] - self.joint_limits_rad[joint_idx, 0])
        
        # Map to servo range
        servo_angle = self.servo_min[joint_idx] + normalized * (self.servo_max[joint_idx] - self.servo_min[joint_idx])
        
        return int(servo_angle)
    
    def joint_commands_callback(self, msg):
        """
        Callback for joint commands
        Sends commands to Arduino
        
        Args:
            msg: JointState message
        """
        if not self.enable_hardware or self.emergency_stop:
            return
        
        if len(msg.position) < 6:
            return
        
        # Convert to servo angles
        servo_angles = []
        for i in range(6):
            servo_angle = self.rad_to_servo_angle(i, msg.position[i])
            servo_angles.append(servo_angle)
        
        # Send to Arduino
        self.send_to_arduino(servo_angles)
        
        # Publish hardware status
        status_msg = Bool()
        status_msg.data = True
        self.hardware_status_pub.publish(status_msg)
    
    def send_to_arduino(self, servo_angles):
        """
        Send servo angles to Arduino via serial
        
        Protocol: Start byte (0xFF) + 6 servo angles (uint8) + checksum
        
        Args:
            servo_angles: List of 6 servo angles (0-180 degrees)
        """
        if self.ser is None or not self.ser.is_open:
            return
        
        try:
            # Create packet
            packet = bytearray()
            packet.append(0xFF)  # Start byte
            
            for angle in servo_angles:
                packet.append(int(angle) & 0xFF)
            
            # Calculate checksum (simple sum)
            checksum = sum(servo_angles) & 0xFF
            packet.append(checksum)
            
            # Send
            self.ser.write(packet)
            
        except Exception as e:
            rospy.logerr(f"Failed to send to Arduino: {e}")
            self.emergency_stop = True
    
    def emergency_stop_callback(self, msg):
        """
        Emergency stop callback
        """
        self.emergency_stop = msg.data
        if self.emergency_stop:
            rospy.logwarn("EMERGENCY STOP ACTIVATED")
    
    def run(self):
        """
        Main loop
        """
        rospy.loginfo("Hardware Bridge running...")
        rospy.spin()
        
        # Cleanup
        if self.ser is not None and self.ser.is_open:
            self.ser.close()


if __name__ == '__main__':
    try:
        node = HardwareBridgeNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
