#!/usr/bin/env python3
import pickle
import random
import math
import rclpy

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tutorial_interfaces.action import Rotate

import numpy as np
import time

import threading
import time


from rclpy.action import ActionServer, CancelResponse, GoalResponse, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup


data_folder = "~/Repos/bitirme/ros2_ws/src/data/"

# TODO: Fix laser min reading problem
# TODO: add 90-degree turns with action server/client

LIDAR_SAMPLE_SIZE = 360
EPISODES = 10000

ANGULAR_VELOCITY = 1.8
ANGULAR_VELOCITY = 18
LINEAR_VELOCITY = 0.8

REAL_TIME_FACTOR = 1


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
actions = ['FORWARD', "LEFT", "RIGHT"]


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
        return
        with open(data_folder + 'Q-Table_v2.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self):
        read_dictionary = {}
        return read_dictionary

        try:
            with open(data_folder + 'Q-Table_v2.pkl', 'rb') as f:
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

        self.rotate_ = ActionClient(
            self, Rotate, "rotate", callback_group=ReentrantCallbackGroup())

        self.laser_ranges = np.zeros(self.lidar_sample_size)
        self.odom_data = Odometry()
        self.previous_action = (0.0, 0.0)

        self.timer_ = self.create_timer(1.0, self.step_callback)

    def execute_move_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        rotation_request = goal_handle.request.rotation
        translation_request = goal_handle.request.translation

        curr_orientation = Utils.euler_from_quaternion(
            self.odom_data.pose.pose.orientation)
        curr_position = self.odom_data.pose.pose.position.x

        target_orientation = curr_orientation + rotation_request
        target_position = curr_position + translation_request

        self.get_logger().info(
            f"curr_pos: {curr_position}, target_pos: {target_position}")

        remaining_translation = target_position - curr_position
        remaining_orientation = target_orientation - curr_orientation

        # while remaining_translation > 0.1 or target_position != curr_position:
        while remaining_translation > 0.1 or remaining_orientation > 0.1:
            # self.get_logger().info(f"Target not reached publishing to /cmd_vel topic...")

            remaining_translation = target_position - curr_position
            remaining_orientation = target_orientation - curr_orientation

            self.get_logger().info(
                f"curr_pos: {curr_position}, target_pos: {target_position}")
            self.get_logger().info(
                f"Remaining translation: {remaining_translation}")

            cmd_vel = Twist()
            cmd_vel.linear.x = 0.8 if remaining_translation > 0.1 else 0.0
            cmd_vel.angular.z = 0.8 if remaining_orientation > 0.1 else 0.0

            print(cmd_vel)

            self.cmd_vel_pub_.publish(cmd_vel)

            curr_orientation = Utils.euler_from_quaternion(
                self.odom_data.pose.pose.orientation)
            curr_position = self.odom_data.pose.pose.position.x

            time.sleep(0.05)

        # stop the movement
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        self.cmd_vel_pub_.publish(stop_cmd)

        goal_handle.succeed()
        result = Move.Result()
        result.success = True

        return result

    def step_callback(self):
        if self.episode_index >= self.episodes:
            self.timer_.cancel()
            self.get_logger().info(
                f"Training completed.\n Q-table: {self.q_learning.q_table}")

            self.q_learning.save_q_table()
            return

        if self.episode_index % 100 == 0:
            self.get_logger().info(f"Saving Q-Table")
            self.q_learning.save_q_table()

        state = self.get_state()
        action = self.q_learning.choose_action(state)

        self.get_logger().info(f"Taking action...")

        # sleeps for 1 second in this method
        self.take_action(action)

        self.get_logger().info(f"Action taken, observing next state...")

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

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.sequence))
        rclpy.shutdown()

    def take_action(self, action):
        msg = Twist()

        if action == "LEFT" or action == "RIGHT":
            self.get_logger().info(f"Turning {action.lower()}")
            # Angular movement
            goal = Rotate.Goal()
            goal.rotation = math.pi / 2 if action == "LEFT" else -math.pi / 2
            self.unpause_physics()

            while not self.rotate_.server_is_ready():
                self.get_logger().info('Rotate action server not available, waiting again...')
            try:
                future = self.rotate_.send_goal_async(goal)
                future.add_done_callback(self.goal_response_callback)

                # Spin until the future is complete
                rclpy.spin_until_future_complete(
                    self, future, self.executor, timeout_sec=1.0)

                # Check if the goal was accepted and succeeded
#                 goal_handle = future.result()
#                 if not goal_handle.accepted:
#                     self.get_logger().error('Goal was rejected by the action server.')
#                     return
#
#                 # Get the result of the action
#                 result_future = goal_handle.get_result_async()
#                 rclpy.spin_until_future_complete(self, result_future)
#                 result = result_future.result()
#
#                 if result.success:
#                     self.get_logger().info('Rotation completed successfully.')
#                 else:
#                     self.get_logger().error('Rotation failed.')

            except Exception as e:
                self.get_logger().error(
                    f"/rotate action call failed: {str(e)}")

        elif action == "FORWARD":
            # Linear movement
            msg = Twist()
            msg.linear.x = LINEAR_VELOCITY

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


class RotateActionServer(Node):
    """Minimal action server that processes one goal at a time."""

    def __init__(self, robot_controller: RobotController):
        super().__init__('rotate_action_server')

        self.get_logger().info("Initializing RotateActionServer")

        self.robot_controller = robot_controller
        self.cmd_vel_pub_ = robot_controller.cmd_vel_pub_
        self._goal_handle = None
        self._goal_lock = threading.Lock()

        self._odom_lock = threading.Lock()  # Add a lock for accessing odom_data

        self.create_subscription(Odometry, "odom", self.odom_callback, 10)
        self.odom_data = Odometry()

        self._action_server = ActionServer(
            self, Rotate, "rotate",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            handle_accepted_callback=self.handle_accepted_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup())

    def odom_callback(self, msg: Odometry):
        # self.get_logger().info("Updating odometry data...")
        self.odom_data = msg

    def destroy(self):
        self.get_logger().info("destroying MoveActionServer")
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""

        self.get_logger().info("Goal callback")

        rotation_request = goal_request.rotation

        self.get_logger().info(f"rotation: {rotation_request} ")

        # return GoalResponse.REJECT

        self.get_logger().info('Accepted goal request')
        return GoalResponse.ACCEPT

    def handle_accepted_callback(self, goal_handle):
        self.get_logger().info("handle_accepted callback")

        with self._goal_lock:
            # This server only allows one goal at a time
            if self._goal_handle is not None and self._goal_handle.is_active:
                self.get_logger().info('Aborting previous goal')
                # Abort the existing goal
                self._goal_handle.abort()
            self._goal_handle = goal_handle

        goal_handle.execute()

    def cancel_callback(self, goal):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')

        rotation_request = goal_handle.request.rotation

        with self._odom_lock:
            curr_orientation = Utils.euler_from_quaternion(
                self.robot_controller.odom_data.pose.pose.orientation)

        curr_orientation = Utils.euler_from_quaternion(
            self.odom_data.pose.pose.orientation)

        target_orientation = curr_orientation + rotation_request
        # Normalize the target orientation to be within -pi to pi
        target_orientation = (target_orientation +
                              math.pi) % (2 * math.pi) - math.pi

        remaining_orientation = target_orientation - curr_orientation

        # feedback
        feedback_msg = Rotate.Feedback()
        feedback_msg.remaining_rotation = remaining_orientation

        Kp = 2.0  # Proportional gain

        # Start executing the action
        # Use a smaller threshold for better precision
        while abs(remaining_orientation) > 0.005:
            # If goal is flagged as no longer active (ie. another goal was accepted),
            # then stop executing
            if not goal_handle.is_active:
                self.get_logger().info('Goal aborted')
                result = Rotate.Result()
                result.success = False
                return result

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                result = Rotate.Result()
                result.success = False
                return result

            with self._odom_lock:
                curr_orientation = Utils.euler_from_quaternion(
                    self.robot_controller.odom_data.pose.pose.orientation)
                remaining_orientation = target_orientation - curr_orientation

            curr_orientation = Utils.euler_from_quaternion(
                self.odom_data.pose.pose.orientation)
            remaining_orientation = target_orientation - curr_orientation

            # Normalize the remaining orientation
            remaining_orientation = (
                remaining_orientation + math.pi) % (2 * math.pi) - math.pi

            # Calculate the angular velocity using a proportional controller
            cmd_vel = Twist()
            cmd_vel.angular.z = Kp * remaining_orientation
            # Cap the angular velocity
            cmd_vel.angular.z = max(
                min(cmd_vel.angular.z, ANGULAR_VELOCITY), -ANGULAR_VELOCITY)

            self.cmd_vel_pub_.publish(cmd_vel)

            # Update feedback
            feedback_msg = Rotate.Feedback()
            feedback_msg.remaining_rotation = round(remaining_orientation, 8)

            # Publish the feedback
            with self._goal_lock:
                self.get_logger().info(
                    'Publishing feedback: {0}'.format(feedback_msg))
                goal_handle.publish_feedback(feedback_msg)

            # Sleep for demonstration purposes
            time.sleep(0.02)

        with self._goal_lock:
            if not goal_handle.is_active:
                self.get_logger().info('Goal aborted')
                result = Rotate.Result()
                result.success = False
                return result

            # goal_handle.succeed()

        #  stop the movement
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        self.cmd_vel_pub_.publish(stop_cmd)

        # Populate result message
        goal_handle.succeed()
        result = Rotate.Result()
        result.success = True

        self.get_logger().info('Returning result: {0}'.format(result.success))

        return result


def main(args=None):
    rclpy.init(args=args)

    executor = MultiThreadedExecutor()

    q_learning = QLearning(actions=actions)
    goal_position = (43.618300, -0.503538)

    node = RobotController(q_learning, goal_position,
                           LIDAR_SAMPLE_SIZE, EPISODES)

    rotate_action_server_ = RotateActionServer(robot_controller=node)

    # executor.add_node(node)
    executor.add_node(rotate_action_server_)

    rclpy.spin(node, executor=executor)

    # rclpy.spin(move_action_server_, executor=executor)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
