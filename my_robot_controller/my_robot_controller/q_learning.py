#!/usr/bin/env python3
import pickle
import random
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty


import numpy as np
import time

# TODO: Fix laser min reading problem
# TODO: add 90-degree turns with action server/client

LIDAR_SAMPLE_SIZE = 360
EPISODES = 10000

ANGULAR_VELOCITY = 1.8
LINEAR_VELOCITY = 0.8

REAL_TIME_FACTOR = 10


# bounds
# x [-10, 47]  y: -19 19
bounds = ((-10, 47), (-19, 19))
x_grid_size = bounds[0][1] - bounds[0][0]  # Define the grid size
y_grid_size = bounds[1][1] - bounds[1][0]  # Define the grid size

# Actions (forward, backward, left, right)
actions = [(-0.5, 0.0), (0.5, 0.0), (0.0, 0.5), (0.0, -0.5)]

# no backward motion
actions = [(LINEAR_VELOCITY, 0.0), (0.0, ANGULAR_VELOCITY),
           (0.0, -ANGULAR_VELOCITY)]
# actions = [(0.0, ANGULAR_VELOCITY), (0.0, -ANGULAR_VELOCITY)]


episode_index = 0


class QLearning:
    def __init__(self, actions, alpha=0.1, gamma=0.9):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = self.load_q_table()

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}

        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        new_q = ((1.0 - self.alpha) * current_q) + \
            (self.alpha * (reward + (self.gamma * max_next_q)))
        self.q_table[state][action] = new_q
        print(f"Updated Q-table with reward: {reward}, Q: {new_q}")

    def choose_action(self, state, epsilon=0.1):
        if random.uniform(0, 1) < epsilon or state not in self.q_table:
            print("choosing random action")
            return random.choice(self.actions)

        print(
            f"Choosing from Q-Table. {max(self.q_table[state], key=self.q_table[state].get)}")
        return max(self.q_table[state], key=self.q_table[state].get)

    def save_q_table(self):
        with open('Q-Table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self):
        read_dictionary = {}

        try:
            with open('Q-Table.pkl', 'rb') as f:
                read_dictionary = pickle.load(f)
            print("Loaded Q-Table")
        except:
            read_dictionary = {}
            print("Q-Table not found")

        return read_dictionary


class Utils:
    @staticmethod
    def discretize(value, min_value, max_value, num_bins):
        return int((value - min_value) / (max_value - min_value) * num_bins)

    @staticmethod
    def get_distance_to_goal(robot_position, goal_position):
        return math.sqrt((goal_position[0] - robot_position[0]) ** 2 + (goal_position[1] - robot_position[1]) ** 2)

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
    def discretize_odom_data(odom, bounds, x_grid_size, y_grid_size):
        """
        Discretizes the odometry data into a discrete state.

        Args:
        - odom: The odometry message containing the position.
        - bounds: A tuple ((min_x, max_x), (min_y, max_y)) representing the bounds of the environment.
        - grid_size: The number of discrete steps in the grid.

        Returns:
        - A tuple (discrete_x, discrete_y) representing the discrete state.
        """
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y

        discrete_x = Utils.discretize_position(
            x, bounds[0], x_grid_size)
        discrete_y = Utils.discretize_position(
            y, bounds[1], y_grid_size)
        return (discrete_x, discrete_y)

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
OBSTACLE_COLLISION_THRESHOLD = 0.4


class RobotController(Node):
    def __init__(self, q_learning: QLearning, goal_position, lidar_sample_size=360, episodes=10000):
        super().__init__("robot_controller")
        self.q_learning = q_learning
        self.goal_position = goal_position
        self.lidar_sample_size = lidar_sample_size
        self.episodes = episodes
        self.episode_index = 0

        self.cmd_vel_pub_ = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(LaserScan, "scan", self.scan_callback, 1)
        self.create_subscription(Odometry, "odom", self.odom_callback, 10)

        self.unpause_ = self.create_client(Empty, "/unpause_physics")
        self.pause_ = self.create_client(Empty, "/pause_physics")

#         self.goal_rotation = self.create_service(
#             Twist, "rotate", self.goal_rotation_service_callback)
#
#         self.create_

        self.laser_ranges = np.zeros(self.lidar_sample_size)
        self.odom_data = Odometry()
        self.previous_action = (0.0, 0.0)

        self.timer_ = self.create_timer(1.0, self.step_callback)

    def goal_rotation_service_callback(self, msg: Twist):

        pass

    def step_callback(self):
        if self.episode_index >= self.episodes:
            self.timer_.cancel()
            self.get_logger().info(
                f"Training completed.\n Q-table: {self.q_learning.q_table}")

            self.q_learning.save_q_table()
            return

        state = self.get_state()
        action = self.q_learning.choose_action(state)

        # sleeps for 1 second in this method
        self.take_action(action)

        # observe the new state
        next_state = self.get_state()
        done = self.check_done()
        reward = self.get_reward(done)
        self.q_learning.update_q_table(state, action, reward, next_state)
        self.episode_index += 1

    def unpause_physics(self):
        while not self.unpause_.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.unpause_.call_async(Empty.Request())
        except:
            self.get_logger().error("/unpause_physics service call failed")

    def pause_physics(self):
        while not self.pause_.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.pause_.call_async(Empty.Request())
        except:
            self.get_logger().error("/gazebo/pause_physics service call failed")

    def take_action(self, action):
        msg = Twist()
        linear_x, angular_z = action
        msg.linear.x = linear_x
        msg.angular.z = angular_z

        if linear_x == 0.0 and abs(angular_z) == ANGULAR_VELOCITY:
            self.cmd_vel_pub_.publish(msg)
            self.unpause_physics()
            time.sleep(1.0 / REAL_TIME_FACTOR)
            stabilize_msg = Twist()
            stabilize_msg.linear.x = 0.0
            stabilize_msg.angular.z = 0.0
            self.cmd_vel_pub_.publish(stabilize_msg)
            time.sleep(0.7 / REAL_TIME_FACTOR)
        else:
            self.cmd_vel_pub_.publish(msg)
            self.unpause_physics()
            time.sleep(1 / REAL_TIME_FACTOR)

        self.previous_action = action

        self.pause_physics()

    def get_state(self):
        x = self.odom_data.pose.pose.position.x
        y = self.odom_data.pose.pose.position.y
        robot_position = Utils.discretize_odom_data(
            self.odom_data, bounds, x_grid_size, y_grid_size)
        orientation = self.odom_data.pose.pose.orientation
        robot_orientation = Utils.euler_from_quaternion(orientation)
        distance_to_goal = Utils.get_distance_to_goal(
            robot_position, self.goal_position)
        angle_to_goal = Utils.get_angle_to_goal(
            robot_position, robot_orientation, self.goal_position)
        distance_to_goal_disc = Utils.discretize(distance_to_goal, 0, 50, 10)
        angle_to_goal_disc = Utils.discretize(
            angle_to_goal, -math.pi, math.pi, 10)

        # state = tuple(self.laser_ranges) + \
        #     (distance_to_goal_disc, angle_to_goal_disc)

        min_distances = Utils.get_min_distances_from_slices(
            self.laser_ranges, 16)

        # Do not include ranges in state
        state = tuple(robot_position) + \
            (distance_to_goal_disc, angle_to_goal_disc) + tuple(min_distances)

        print(f"State: {state}")
        return state

    def check_done(self):
        if Utils.get_distance_to_goal((self.odom_data.pose.pose.position.x, self.odom_data.pose.pose.position.y), self.goal_position) < GOAL_REACHED_THRESHOLD:
            self.get_logger().info(
                f"Goal reached. Distance to goal: {Utils.get_distance_to_goal((self.odom_data.pose.pose.position.x, self.odom_data.pose.pose.position.y), self.goal_position)}")
            return True

        if min(self.laser_ranges) < OBSTACLE_COLLISION_THRESHOLD:
            self.get_logger().info(
                f"Collision detected. minRange: {min(self.laser_ranges)}")
            return True

        return False

    def get_reward(self, done):
        distance_to_goal = Utils.get_distance_to_goal(
            (self.odom_data.pose.pose.position.x, self.odom_data.pose.pose.position.y), self.goal_position)

        reached_goal = distance_to_goal < GOAL_REACHED_THRESHOLD
        collision = min(self.laser_ranges) < OBSTACLE_COLLISION_THRESHOLD

        if done:
            if reached_goal:
                return 100
            if collision:
                return -100

        distance_reward = (1/distance_to_goal) * 10
        step_penalty = -1

        reward = distance_reward - step_penalty

        return reward

    def scan_callback(self, msg: LaserScan):
        self.laser_ranges = msg.ranges

    def odom_callback(self, msg: Odometry):
        self.odom_data = msg


def main(args=None):
    rclpy.init(args=args)

    q_learning = QLearning(actions=actions)
    goal_position = (43.618300, -0.503538)

    node = RobotController(q_learning, goal_position,
                           LIDAR_SAMPLE_SIZE, EPISODES)

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
