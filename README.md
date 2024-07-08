# Swarm Management with Reinforcement Learning

## Introduction

This project implements swarm management using Reinforcement Learning, specifically the TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm. It demonstrates the use of deep reinforcement learning techniques to control a swarm of robots in a simulated environment. Using ROS 2 and Gazebo, the project simulates multiple agents (three in this case) navigating to a goal position while avoiding obstacles.

## Prerequisites

- ROS 2 Humble
- Gazebo Classic 11.10.2
- Python 3.10.12
- PyTorch 2.3.1
- NumPy 1.21.5
- Matplotlib 3.5.1
- TensorBoard

## Installation

1. Install ROS 2 Humble following the [official instructions](https://docs.ros.org/en/humble/Installation.html).

2. Install Colcon following the [official instructions](https://colcon.readthedocs.io/en/released/user/installation.html).

3. Install Gazebo packages:
```bash
sudo apt install ros-humble-gazebo-ros-pkgs
```

4. Install Xacro:
```bash
sudo apt install ros-humble-xacro 
```

5. Create a ROS 2 workspace:
```bash
mkdir -p ~/ros2_ws/
cd ~/ros2_ws/
```

6. Clone this repository:
```bash
git clone https://github.com/kgndnc/ros-gazebo-deep-rl.git
```

7. Rename the folder
```bash
mv ros-gazebo-deep-rl/ src/
```

8. Build the ROS 2 packages:
```bash
cd ~/ros2_ws
colcon build
```

## Usage

Source the ROS 2 setup files:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

### For testing
1. Launch the Gazebo simulation:
```bash
ros2 launch my_robot_bringup test_my_robot_gazebo.launch.xml 
```

2. In a new terminal, run the test script:
```bash
cd ~/ros2_ws/src/my_robot_controller/my_robot_controller/td3/
python3 test.py
```

### For training
1. Launch the Gazebo simulation:
```bash
ros2 launch my_robot_bringup my_robot_gazebo.launch.xml 
```

2. In a new terminal, run the training script:
```bash
cd ~/ros2_ws/src/my_robot_controller/my_robot_controller/td3/
python3 train.py
```

## Project Structure

- `my_robot_controller/`: ROS 2 package containing nodes and Python scripts that implement RL algorithms
- `my_robot_bringup/`: ROS 2 package containing launch files and world descriptions
  - `launch/`: Launch files for starting the Gazebo simulation
  - `worlds/`: Gazebo world files in SDF format

- `my_robot_description/`: ROS 2 package for robot model descriptions
  - `urdf/`: Robot model files in URDF format


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
