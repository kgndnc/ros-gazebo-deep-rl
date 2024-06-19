#!/usr/bin/env python3
import os
import sys
from concurrent.futures import Future
import pickle
import random
import math
import rclpy

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
import rclpy.waitable
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tutorial_interfaces.action import Rotate

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from datetime import timedelta

import threading


matplotlib.use('agg')


data_folder = os.path.expanduser("~/Repos/bitirme/ros2_ws/src/data/")
data_name = "Q-Table_small_world_w_rooms_v5"
# Version notes
# v5: updated reward strategy


LIDAR_SAMPLE_SIZE = 180
EPISODES = 200_000

ANGULAR_VELOCITY = 1.8
LINEAR_VELOCITY = 0.9
REAL_TIME_FACTOR = 10


# for every ... episode save to file
SAVE_INTERVAL = 1000


# bounds
# x [-10, 47]  y: -19 19
# my_world.sdf
bounds = ((-10, 47), (-19, 19))
# small_world.sdf
bounds = ((-10, 10), (-10, 14))


x_grid_size = bounds[0][1] - bounds[0][0]  # Define the grid size
y_grid_size = bounds[1][1] - bounds[1][0]  # Define the grid size

# hypotenuse of the environment - radius of the robot
max_distance_to_goal = math.floor(
    math.sqrt(x_grid_size**2 + y_grid_size**2) - 0.6)


actions = ['FORWARD', "LEFT", "RIGHT", "STAY"]


# global variables for sensor data
agent_count = 3
laser_ranges = np.array([np.zeros(LIDAR_SAMPLE_SIZE)
                        for _ in range(agent_count)])


epsilon_discount = 0.99996
epsilon_discount = 0.999986


odom_data = np.array([Odometry() for _ in range(agent_count)])

# print(f"laser_ranges: {laser_ranges}")
# print(f"odom_data: {odom_data}")

#  To break ties I chose "Do Nothing" because that is the more common action


class QLearning:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.9, _save_q_table=True, _load_q_table=True):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self._save_q_table = _save_q_table
        self._load_q_table = _load_q_table
        self.q_table = self.load_q_table()
        self.lock = threading.Lock()  # Create a threading lock

    def update_q_table(self, state, action, reward, next_state):
        with self.lock:
            if state not in self.q_table:
                self.q_table[state] = {a: 0 for a in self.actions}
            if next_state not in self.q_table:
                self.q_table[next_state] = {a: 0 for a in self.actions}

            current_q = self.q_table[state][action]
            max_next_q = max(self.q_table[next_state].values())
            new_q = ((1.0 - self.alpha) * current_q) + \
                (self.alpha * (reward + (self.gamma * max_next_q)))
            self.q_table[state][action] = new_q
            # print(f"Updated Q-table with reward: {reward}, Q: {new_q}")

    def choose_action(self, state, episode_index: int):

        if random.uniform(0, 1) < self.epsilon or state not in self.q_table:
            action = random.choice(self.actions)
            # print("choosing random action: " + action)
            return action
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)
            # print(
            #     f"Choosing from Q-Table. {action}")
            return action

    def save_q_table(self):
        if not self._save_q_table:
            return

        with self.lock:  # Acquire the lock
            with open(data_folder + data_name + '.pkl', 'wb') as f:
                pickle.dump(self.q_table, f)
                print(f"Q-Table saved to file {f.name}")
                RobotController.done = False  # may cause a bug
            # Release the lock automatically when exiting the with statement

    def load_q_table(self):
        q_table = {}

        if not self._load_q_table:
            return q_table

        try:
            with open(data_folder + data_name + '.pkl', 'rb') as f:
                q_table = pickle.load(f)
            print("Loaded Q-Table")
        except:
            q_table = {}
            print("Q-Table not found")

        return q_table


class Utils:
    @staticmethod
    def discretize(value, min_value, max_value, num_bins):
        return int((value - min_value) / (max_value - min_value) * num_bins)

    @staticmethod
    def get_angle_between_points(ref_point, point_1_heading, target_point):

        target_vector = [target_point[0] - ref_point[0],
                         target_point[1] - ref_point[1]]

        target_angle = math.atan2(target_vector[1], target_vector[0])

        angle = target_angle - point_1_heading
        return angle

    @staticmethod
    def get_distance_between_points(point_1: tuple[float, float], point_2: tuple[float, float]):
        x_1, y_1 = point_1
        x_2, y_2 = point_2

        dist = math.sqrt(((y_2 - y_1)**2) + ((x_2 - x_1)**2))
        return dist

    @staticmethod
    def get_distance_to_goal(robot_position, goal_position):
        return Utils.get_distance_between_points(robot_position, goal_position)

    @staticmethod
    def get_angle_to_goal(robot_position, robot_orientation, goal_position):
        goal_vector = [goal_position[0] - robot_position[0],
                       goal_position[1] - robot_position[1]]
        goal_angle = math.atan2(goal_vector[1], goal_vector[0])

        # Assuming robot_orientation is given as yaw angle (heading)
        angle_to_goal = goal_angle - robot_orientation
        return angle_to_goal

    @staticmethod
    def euler_from_quaternion(quaternion):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        """
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return yaw_z  # in radians

    @staticmethod
    def discretize_position(position, bounds, grid_size):
        """
        Discretizes a continuous position into a grid index.

        Args:
        - position: The continuous position value (x or y).
        - bounds: A tuple (min_value, max_value) representing the bounds of the environment.
        - grid_size: The number of discrete steps in the grid.

        Returns:
        - The discrete index corresponding to the position.
        """
        min_value, max_value = bounds
        scale = grid_size / (max_value - min_value)
        index = int((position - min_value) * scale)
        # Ensure the index is within bounds
        index = max(0, min(grid_size - 1, index))

        return index

    @staticmethod
    def get_position_from_odom_data(odom):

        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y

        # discrete_x = Utils.discretize_position(
        #     x, bounds[0], x_grid_size*2)
        # discrete_y = Utils.discretize_position(
        #     y, bounds[1], y_grid_size*2)
        return (x, y)

    @staticmethod
    def get_min_distances_from_slices(laser_data, num_slices=4):
        """
        Divide the laser data into slices and take the minimum distance from each slice.

        Args:
        - laser_data: Array of laser scan distances.
        - num_slices: Number of slices to divide the laser data into (default is 4).

        Returns:
        - List of minimum distances from each slice.
        """
        slice_size = len(laser_data) // num_slices
        min_distances = []

        for i in range(num_slices):
            start_index = i * slice_size
            end_index = start_index + slice_size
            slice_min = min(laser_data[start_index:end_index])
            # slice_min = round(slice_min, 2)
            slice_min = round(slice_min, 0)
            min_distances.append(slice_min)

        return min_distances


GOAL_REACHED_THRESHOLD = 1.0
OBSTACLE_COLLISION_THRESHOLD = 0.5


class RobotController(Node):
    episode_index = 0
    done = False
    is_physics_paused = False

    def __init__(self, q_learning: QLearning, goal_position, namespace: str, robot_index: int, lidar_sample_size=360, episodes=10000, episode_index=0):
        # super().__init__("robot_controller_" + namespace)
        super().__init__("robot_controller_" + namespace, namespace=namespace)
        RobotController.episode_index = episode_index

        self.q_learning = q_learning
        self.goal_position = goal_position
        self.lidar_sample_size = lidar_sample_size
        self.episodes = episodes

        self.robot_index = robot_index
        self.first_agent = robot_index == 0
        self.last_agent = robot_index+1 == agent_count
        self.step_counter = 0  # Add a step counter

        self.cmd_vel_pub_ = self.create_publisher(
            Twist, "/" + namespace + "/cmd_vel", 10)

        self.unpause_ = self.create_client(Empty, "/unpause_physics")
        self.pause_ = self.create_client(Empty, "/pause_physics")
        self.reset_ = self.create_client(Empty, "/reset_world")

        self.total_reward_during_episode = 0
        self.episode_rewards = []

        self.rotation_active = False
        self.target_orientation = None
        self.rotate_timer_ = None
        self.Kp = 1.3

        self.start_time = time.time()
        self.timer_ = self.create_timer(2.5 / REAL_TIME_FACTOR, self.step)
        # self.timer_ = self.create_timer(1.5 / REAL_TIME_FACTOR, self.step)

        # Initialize previous distance to goal
        # self.prev_distance_to_goal = Utils.get_distance_to_goal(
        #     self.get_robot_position(), self.goal_position)

        self.last_actions = {agent_index: (0.0, 0.0)
                             for agent_index in range(agent_count)}
        self.prev_distances_to_goal = [Utils.get_distance_to_goal(self.get_robot_position(
            agent_index), self.goal_position) for agent_index in range(agent_count)]

    def step(self):
        # self.get_logger().info(f"Inside step function")

        if RobotController.done:
            return  # If any agent is done, do nothing

        if RobotController.episode_index >= self.episodes:
            self.finalize_training()
            self.executor.shutdown()
            return

        if (RobotController.episode_index) % SAVE_INTERVAL == 0 and self.last_agent:
            RobotController.done = True  # may cause a bug
            self.get_logger().info(f"Saving Q-Table")
            RobotController.episode_index += 1
            self.q_learning.save_q_table()

        state = self.get_state()
        action = self.q_learning.choose_action(
            state, RobotController.episode_index)

        # self.get_logger().info(f"Taking action...")

        # sleeps for a duration in this method
        self.take_action(action)

        # observe the new state
        next_state = self.get_state()
        RobotController.done = self.check_done()
        reward = self.get_reward(RobotController.done, action)
        self.total_reward_during_episode += reward
        self.q_learning.update_q_table(state, action, reward, next_state)
        self.step_counter += 1

        if RobotController.done:
            self.reset_environment()
            return

        if self.step_counter >= 1000:  # End episode if step limit is reached
            RobotController.done = True
            self.get_logger().info(f"Step limit reached. Ending episode.")
            self.reset_environment()
            return

        # self.get_logger().info(f"Episode: {RobotController.episode_index}")

    def reset_environment(self):

        global odom_data
        global laser_ranges

        self.step_counter = 0  # Reset step counter

        # Reset robot positions or other necessary environment variables
        # Example: Reset robot position, reset goal position, etc.
        # You might need to call a ROS service to reset the simulation environment

        future = self.reset_.call_async(Empty.Request())

        def future_complete(future: Future):
            msg = future.result()

        future.add_done_callback(future_complete)

        for i in range(agent_count):
            odom_data[i] = Odometry()  # Reset odom data for all robots
        laser_ranges = np.array([np.zeros(LIDAR_SAMPLE_SIZE)
                                 for _ in range(agent_count)])

        RobotController.done = False  # Reset global done flag
        self.episode_rewards.append(self.total_reward_during_episode)
        current_time = time.time() - self.start_time
        print(
            f"EP: {RobotController.episode_index} - [alpha: {self.q_learning.alpha} - gamma: {self.q_learning.gamma} - epsilon: {self.q_learning.epsilon:1.2f}] - Reward: {self.total_reward_during_episode} \tTime: {timedelta(seconds=current_time)}")
        self.total_reward_during_episode = 0
        RobotController.episode_index += 1
        self.q_learning.epsilon *= epsilon_discount

        # Reset the previous distance to the goal
        self.prev_distance_to_goal = Utils.get_distance_to_goal(
            self.get_robot_position(), self.goal_position)

    def take_action(self, action):
        msg = Twist()

        if action == "LEFT" or action == "RIGHT":
            # Angular movement
            # self.get_logger().info(f"Turning {action.lower()}")

            angle = math.pi / 2 if action == "LEFT" else -math.pi / 2

            # time_required = abs(angle / (self.Kp * ANGULAR_VELOCITY))
            time_required = abs(angle / ANGULAR_VELOCITY)

            msg.angular.z = ANGULAR_VELOCITY if action == "LEFT" else -ANGULAR_VELOCITY
            msg.angular.z *= self.Kp
            msg.linear.x = 0.15

            if RobotController.is_physics_paused:
                self.unpause_physics()

            self.cmd_vel_pub_.publish(msg)
            time.sleep(time_required / REAL_TIME_FACTOR)

        elif action == "FORWARD":
            # Linear movement

            msg.linear.x = LINEAR_VELOCITY

            if RobotController.is_physics_paused:
                self.unpause_physics()

            self.cmd_vel_pub_.publish(msg)
            time.sleep(1.0 / REAL_TIME_FACTOR)
        elif action == "STAY":
            time.sleep(1.0 / REAL_TIME_FACTOR)

        self.previous_action = action

        # stop the movement and wait for it to stabilize
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        self.cmd_vel_pub_.publish(stop_cmd)
        time.sleep(0.5 / REAL_TIME_FACTOR)

        # pause physics here
        if not RobotController.is_physics_paused:
            self.pause_physics()

    def get_state(self):

        global odom_data
        global laser_ranges

        robot_position = Utils.get_position_from_odom_data(
            odom_data[self.robot_index])
        orientation = odom_data[self.robot_index].pose.pose.orientation
        robot_orientation = Utils.euler_from_quaternion(orientation)
        distance_to_goal = Utils.get_distance_to_goal(
            robot_position, self.goal_position)

        # if self.last_agent:
        #     print(f"Robot position = {robot_position}")

        angle_to_goal = Utils.get_angle_to_goal(
            robot_position, robot_orientation, self.goal_position)
        # 18 artırılabilir
        distance_to_goal_disc = Utils.discretize(
            distance_to_goal, 0, max_distance_to_goal, 18)
        angle_to_goal_disc = Utils.discretize(
            angle_to_goal, -math.pi, math.pi, 18)
        robot_orientation_disc = Utils.discretize(
            robot_orientation, -math.pi, math.pi, 8)

        # state = tuple(laser_ranges[self.robot_index]) + \
        #     (distance_to_goal_disc, angle_to_goal_disc)

        # min_distances = Utils.get_min_distances_from_slices(
        #     laser_ranges[self.robot_index], 16)

        collision = True if min(
            laser_ranges[self.robot_index]) < OBSTACLE_COLLISION_THRESHOLD else False

        """
        State:
            - distance to goal
            - relative angle to goal
            - robot's orientation (yaw or heading)
            - collision: boolean
            # - reached to target (not required)
            - for each other robot_i:       (append these to tuple)
                - distance to robot_i
                - relative angle to robot_i

        """

        # old state
        # state = tuple(robot_position) + (robot_orientation_disc,) + \
        #     (distance_to_goal_disc, angle_to_goal_disc) + tuple(min_distances)

        # new state
        # Initialize tuple for information about other robots
        info_about_other_robots = ()

        # Calculate distances and relative angles to other robots
        for i, odom in enumerate(odom_data):
            if i != self.robot_index:  # Skip the current robot
                other_robot_position = Utils.get_position_from_odom_data(odom)

                # Calculate distance to the other robot
                distance_to_robot_i = Utils.get_distance_between_points(
                    robot_position, other_robot_position)

                # Calculate relative angle to the other robot
                angle_to_robot_i = Utils.get_angle_between_points(
                    robot_position, robot_orientation, other_robot_position)

                # Discretize distance and angle
                distance_to_robot_i_disc = Utils.discretize(
                    distance_to_robot_i, 0, max_distance_to_goal, 18)
                angle_to_robot_i_disc = Utils.discretize(
                    angle_to_robot_i, -math.pi, math.pi, 8)

                # Append the information to the tuple
                info_about_other_robots += (distance_to_robot_i_disc,
                                            angle_to_robot_i_disc)

        state = (distance_to_goal_disc, angle_to_goal_disc, robot_orientation_disc,
                 collision) + tuple(info_about_other_robots)

        # if self.last_agent:
        #     print(f"State for {self.robot_index+1}: {state}")

        return state

    def check_done(self):
        if Utils.get_distance_to_goal((odom_data[self.robot_index].pose.pose.position.x, odom_data[self.robot_index].pose.pose.position.y), self.goal_position) < GOAL_REACHED_THRESHOLD:
            self.get_logger().info(
                f"Goal reached. Distance to goal: {Utils.get_distance_to_goal((odom_data[self.robot_index].pose.pose.position.x, odom_data[self.robot_index].pose.pose.position.y), self.goal_position)}")
            return True

        if min(laser_ranges[self.robot_index]) < OBSTACLE_COLLISION_THRESHOLD:
            self.get_logger().info(
                f"Collision detected. minRange: {min(laser_ranges[self.robot_index])}")
            return True

        return False

    def _get_reward(self, done, agent_index: int):
        r_arrive = 200
        r_collision = -200
        k = 5

        distance_to_goal = Utils.get_distance_to_goal(
            (odom_data[agent_index].pose.pose.position.x, odom_data[agent_index].pose.pose.position.y), self.goal_position)

        reached_goal = distance_to_goal < GOAL_REACHED_THRESHOLD
        collision = min(laser_ranges[agent_index]
                        ) < OBSTACLE_COLLISION_THRESHOLD

        if done:
            if reached_goal:
                return r_arrive
            if collision:
                return r_collision

        total_aproach_reward = 0
        for i, _ in enumerate(odom_data):
            current_distance_to_goal = Utils.get_distance_to_goal(
                self.get_robot_position(i), self.goal_position)

            approach_dist = self.prev_distances_to_goal[i] - \
                current_distance_to_goal
            approach_dist *= k

            self.prev_distances_to_goal[i] = current_distance_to_goal

            total_aproach_reward += approach_dist

        self.prev_distances_to_goal = [Utils.get_distance_to_goal(self.get_robot_position(
            agent_index), self.goal_position) for agent_index in range(agent_count)]

        return total_aproach_reward

    def get_reward(self, done, agent_index):
        r_arrive = 200
        r_collision = -200
        k = 5

        distance_to_goal = Utils.get_distance_to_goal(
            (odom_data[self.robot_index].pose.pose.position.x, odom_data[self.robot_index].pose.pose.position.y), self.goal_position)

        reached_goal = distance_to_goal < GOAL_REACHED_THRESHOLD
        collision = min(laser_ranges[self.robot_index]
                        ) < OBSTACLE_COLLISION_THRESHOLD

        if done:
            if reached_goal:
                return r_arrive
            if collision:
                return r_collision

        total_aproach_reward = 0
        for i, _ in enumerate(odom_data):
            current_distance_to_goal = Utils.get_distance_to_goal(
                self.get_robot_position(), self.goal_position)

            approach_dist = self.prev_distances_to_goal[i] - \
                current_distance_to_goal
            approach_dist *= k

            self.prev_distances_to_goal[i] = current_distance_to_goal

            total_aproach_reward += approach_dist

        return total_aproach_reward

    def get_robot_position(self, robot_index=None):
        global odom_data
        if robot_index == None:
            robot_index = self.robot_index
        return Utils.get_position_from_odom_data(odom_data[robot_index])

    def unpause_physics(self):
        return
        while not self.unpause_.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.unpause_.call_async(Empty.Request())
            RobotController.is_physics_paused = False
        except:
            self.get_logger().error("/unpause_physics service call failed")

    def pause_physics(self):
        return
        while not self.pause_.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.pause_.call_async(Empty.Request())
            RobotController.is_physics_paused = True
        except:
            self.get_logger().error("/gazebo/pause_physics service call failed")

    def finalize_training(self):
        self.timer_.cancel()
        self.get_logger().info(
            f"Training completed.\n Q-table: {self.q_learning.q_table}")
        if self.last_agent:
            self.q_learning.save_q_table()

            # Calculate the moving average of rewards
            window_size = 100
            moving_avg_rewards = np.convolve(
                self.episode_rewards, np.ones(window_size)/window_size, mode='valid')

            # Plot the results
            plt.plot(moving_avg_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Moving Average Reward (Window Size = 100)')
            plt.title(f'Learning Progress for Agent {self.robot_index}')
            plt.savefig(data_folder + "figures/reward_" +
                        str(self.robot_index) + '.png')
            plt.close()

            with open(data_folder + "rewards_array" + '.pkl', 'wb') as f:
                pickle.dump(moving_avg_rewards, f)

            with open(data_folder + "rewards.txt", "w") as file:
                # Iterate over the list and write each item to the file
                file.write("[ ")
                for item in moving_avg_rewards:
                    # Convert each item to a string
                    file.write(str(item) + ",")
                file.write(" ]")


class OdomSubscriber(Node):
    def __init__(self, namespace: str, robot_index: int):
        super().__init__('odom_subscriber_' + namespace)
        self.subscription = self.create_subscription(
            Odometry,
            "/" + namespace + '/odom',
            self.odom_callback,
            10)
        self.robot_index = robot_index
        self.subscription

    def odom_callback(self, msg: Odometry):
        global odom_data
        odom_data[self.robot_index] = msg


class ScanSubscriber(Node):
    def __init__(self, namespace: str, robot_index: int):
        super().__init__('scan_subscriber_' + namespace)
        self.subscription = self.create_subscription(
            LaserScan, "/" + namespace + "/scan", self.scan_callback, 10)
        self.robot_index = robot_index
        self.subscription

    def scan_callback(self, msg: LaserScan):
        # self.get_logger().info(f"Updating scan data for {self.robot_index}")
        global laser_ranges
        laser_ranges[self.robot_index] = msg.ranges


def main(args=None):
    rclpy.init(args=args)

    # my_world.sdf
    goal_position = (43.618300, -0.503538)

    # small_world.sdf
    goal_position = (-8.061270, 1.007540)

    namespaces = ["robot_1", "robot_2", "robot_3"]
    executor = MultiThreadedExecutor()

    # if not in training set epsilon as 0.0
    q_learning = QLearning(actions=actions, alpha=0.7, epsilon=0.4,
                           _save_q_table=True, _load_q_table=True)

    for i, namespace in enumerate(namespaces):
        robot_index = i
        robot_controller = RobotController(q_learning, goal_position, namespace, robot_index,
                                           episode_index=11_0020,
                                           lidar_sample_size=LIDAR_SAMPLE_SIZE,
                                           episodes=EPISODES)
        odom_subscriber = OdomSubscriber(namespace, robot_index)
        scan_subscriber = ScanSubscriber(namespace, robot_index)

        executor.add_node(odom_subscriber)
        executor.add_node(scan_subscriber)
        executor.add_node(robot_controller)

    executor_thread = threading.Thread(target=executor.spin, daemon=False)
    # executor_thread.start()
    executor_thread.run()

    # executor.spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
