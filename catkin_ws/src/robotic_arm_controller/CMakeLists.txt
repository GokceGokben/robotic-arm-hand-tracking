cmake_minimum_required(VERSION 3.0.2)
project(robotic_arm_controller)

## Find catkin and required packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  tf
  tf2_ros
  message_generation
  gazebo_ros
)

## Generate messages
add_message_files(
  FILES
  HandPose.msg
  GestureCommand.msg
)

## Generate services (if needed)
# add_service_files(
#   FILES
#   SolveIK.srv
# )

## Generate added messages and services
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

## Catkin package configuration
catkin_package(
  CATKIN_DEPENDS
    rospy
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    message_runtime
)

## Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Install Python scripts
catkin_install_python(PROGRAMS
  scripts/hand_tracker_node.py
  scripts/ik_solver_node.py
  scripts/robot_controller_node.py
  scripts/coordinate_mapper_node.py
  scripts/hardware_bridge_node.py
  scripts/kinematics.py
  scripts/gesture_recognizer.py
  scripts/transform_utils.py
  scripts/visualization.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

## Install URDF files
install(DIRECTORY urdf/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf
)

## Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
