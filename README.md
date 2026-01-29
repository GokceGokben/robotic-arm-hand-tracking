
# Virtual Robotic Arm Controller

This project enables real-time control of a 6-DOF robotic arm using hand tracking and gesture recognition. It supports both Windows (PyBullet simulation) and Linux (ROS/Gazebo integration).

## Demo 

https://github.com/user-attachments/assets/0c9d80a6-7ef1-4c96-ae4d-960721e19498


**Main features:**
* Real-time hand tracking (MediaPipe)
* Gesture-based robot control (fist, open, pinch, peace)
* Inverse kinematics for smooth arm movement
* Windows: PyBullet simulation (no ROS required)
* Linux: Full ROS/Gazebo support

## 🎯 Features

- **Hand Tracking**: Real-time hand tracking using MediaPipe
- **Gesture Recognition**: Control modes via hand gestures (pinch, fist, open palm)
- **Inverse Kinematics**: Both analytical (geometric) and numerical (Jacobian-based) IK solvers
- **D-H Parameters**: Proper Denavit-Hartenberg parameter implementation
- **Cross-Platform**: 
  - **Windows**: PyBullet simulation (native)
  - **Linux**: Full ROS/Gazebo integration
- **Hardware Support**: Arduino/Raspberry Pi integration for physical robot control
- **Real-time Control**: Smooth motion with position filtering and velocity limiting


## 🪟 Windows Quick Start


```bash
# 1. (Recommended) Create a virtual environment
python -m venv .venv
# 2. Activate the environment
#    (Windows)
.venv\Scripts\activate
# 3. Install dependencies
pip install -r requirements_windows.txt
# 4. Run the application
python pybullet_sim/main.py
```

## 📋 Prerequisites

**Windows (PyBullet):**
- Python 3.8–3.12 (MediaPipe is not compatible with Python 3.13+)
- Webcam
- Windows 10/11

**Linux (ROS):**
- ROS Noetic
- Gazebo
- Ubuntu 20.04

**To install ROS dependencies:**
```bash
sudo apt update
sudo apt install ros-noetic-desktop-full
```


## 🎮 Controls

- **Open Palm**: Normal tracking mode (move robot)
- **Pinch**: Precision mode (fine movement)
- **Fist**: Close gripper (hold object, now robot keeps moving)
- **Press 'q'**: Quit application

## 📊 System Architecture

### Windows (PyBullet)
```
Camera → Hand Tracker → Coordinate Mapper → IK Solver → PyBullet Simulation
           ↓
    Gesture Recognizer
```

### Linux (ROS)
```
Camera → hand_tracker_node → coordinate_mapper_node → ik_solver_node → robot_controller_node → Gazebo
                ↓
         gesture_command
```


## 📁 Project Structure

```
Robotic/
├── pybullet_sim/
│   ├── main.py            # Main Windows app
│   ├── robot_controller.py
│   └── hand_tracker.py
├── catkin_ws/
│   └── src/robotic_arm_controller/
│       ├── scripts/
│       ├── launch/
│       ├── urdf/
│       ├── config/
│       ├── msg/
│       └── arduino/
├── requirements_windows.txt
├── requirements.txt
├── README.md
├── run_robot.bat
├── LICENSE
```

## 🔧 Hardware Integration (Optional)

Connect to a physical robot arm using Arduino:

1. Upload `arduino/arduino_controller.ino` to Arduino
2. Connect 6 servo motors
3. **Windows**: Edit `main.py` to enable serial communication
4. **Linux**: Run `roslaunch robotic_arm_controller hardware.launch`

## 🧪 Testing


### Test Kinematics (Linux/ROS)
```bash
python catkin_ws/src/robotic_arm_controller/scripts/test_kinematics.py
```

### Test Components (Windows)
```bash
# Test simulation only
python pybullet_sim/robot_controller.py
# Test hand tracking only
python pybullet_sim/hand_tracker.py
```

## 📚 Documentation

- **D-H Parameters**: See `config/robot_config.yaml`
- **Kinematics**: See `scripts/kinematics.py`

## 🎓 Learning Resources

### Denavit-Hartenberg Parameters
- [Introduction to Robotics - Craig](https://www.pearson.com/us/higher-education/program/Craig-Introduction-to-Robotics-Mechanics-and-Control-4th-Edition/PGM91709.html)
- [Modern Robotics - Lynch & Park](http://hades.mech.northwestern.edu/index.php/Modern_Robotics)

### Inverse Kinematics
- Analytical vs Numerical IK
- Jacobian-based methods
- Optimization approaches

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add MoveIt integration (Linux)
- [ ] Implement trajectory planning
- [ ] Add gripper control
- [ ] Support for different robot configurations
- [ ] Machine learning-based gesture recognition
- [ ] VR/AR visualization


## 🐛 Troubleshooting

### Windows
- **Python 3.13+ not supported**: MediaPipe hand tracking will not work. Use Python 3.8–3.12.
- **Camera not working**: Check camera ID in `hand_tracker.py`.
- **Import errors**: Run `pip install -r requirements_windows.txt`.
- **PyBullet won't start**: Update graphics drivers.

### Linux
- **ROS not found**: Source workspace: `source devel/setup.bash`
- **Gazebo crashes**: `killall gzserver gzclient` and restart
- **Camera permission**: `sudo chmod 666 /dev/video0`


## 🙏 Acknowledgments

- **MediaPipe** by Google for hand tracking
- **PyBullet** for physics simulation
- **ROS** community for robotics framework
- **Gazebo** for advanced simulation

---

**Built with ❤️ for robotics education and research**

**Platform**: Windows (PyBullet) | Linux (ROS/Gazebo)
