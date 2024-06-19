#!/usr/bin/env python3


import logging
import threading
import time
from Utils import Tensorboard
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from gymnasium import spaces
from gymnasium import Env
import rclpy
from sensor_msgs.msg import LaserScan
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import math
# import keras._tf_keras
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.layers import Dense

import keras._tf_keras.keras as keras

from typing import Any, SupportsFloat
import matplotlib
import matplotlib.pyplot as plt
import os
import tensorflow as tf


TOTAL_EPISODES = 10_000


matplotlib.use('agg')


data_folder = os.path.expanduser("~/Repos/bitirme/ros2_ws/src/data/")
data_name = "ddpg"


LIDAR_SAMPLE_SIZE = 180

ANGULAR_VELOCITY = 1.8
LINEAR_VELOCITY = 0.9
REAL_TIME_FACTOR = 77


# for every ... episode save weights
SAVE_INTERVAL = 5


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
max_distance_to_goal *= 1.0


# global variables for sensor data
agent_count = 1
laser_ranges = np.array([np.zeros(LIDAR_SAMPLE_SIZE)
                        for _ in range(agent_count)])


epsilon_discount = 0.99996
epsilon_discount = 0.999986


odom_data = np.array([Odometry() for _ in range(agent_count)])

# print(f"laser_ranges: {laser_ranges}")
# print(f"odom_data: {odom_data}")


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512,
                 name='critic', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name+'_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q


class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_actions=2, name='actor',
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name+'_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)

        return mu


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None,
                 gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 fc1=184, fc2=184, batch_size=300, noise=0.5):
        print(f"input_dims", input_dims)
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.max_action_2 = env.action_space.high[1]
        self.min_action_2 = env.action_space.low[1]

        print(f"max_action = {self.max_action}")
        print(f"min_action = {self.min_action}")
        print(f"max_action_2 = {self.max_action_2}")
        print(f"min_action_2 = {self.min_action_2}")

        self.actor = ActorNetwork(
            fc1_dims=fc1, fc2_dims=fc2, n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2, name='critic')
        self.target_actor = ActorNetwork(fc1_dims=fc1, fc2_dims=fc2,  n_actions=n_actions,
                                         name='target_actor')
        self.target_critic = CriticNetwork(
            fc1_dims=fc1, fc2_dims=fc2, name='target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def __choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)

        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions],
                                        mean=0.0, stddev=self.noise)

        # Scale the actions back to the original action space
        actions = actions.numpy()[0]
        actions[0] = actions[0] * (self.max_action - self.min_action) / \
            2 + (self.max_action + self.min_action) / 2
        actions[1] = actions[1] * (self.max_action_2 - self.min_action_2) / \
            2 + (self.max_action_2 + self.min_action_2) / 2

        # Clip the actions to be within the action space bounds
        actions = np.clip(actions, [self.min_action, self.min_action_2], [
                          self.max_action, self.max_action_2])

        print("actions", actions)

        return actions

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        print("is there a problem??")

        actions = self.actor(state)
        print("actions", actions)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions],
                                        mean=0.0, stddev=self.noise)
        # note that if the env has an action > 1, we have to multiply by
        # max action at some point
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(
                states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()


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
        - grid_size: The number of discretes in the grid.

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
OBSTACLE_COLLISION_THRESHOLD = 0.7


state_dim = 362  # 360 lidar ranges + 2 odom position + 2 odom orientation
state_dim = 180 + 1 + 1
action_dim = 2  # linear velocity and angular velocity
max_action = 1.0
min_action = -1.0


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
        # print(f"Updating odom data")


class ScanSubscriber(Node):
    def __init__(self, namespace: str, robot_index: int):
        super().__init__('scan_subscriber_' + namespace)
        self.subscription = self.create_subscription(
            LaserScan, "/" + namespace + "/scan", self.scan_callback, 10)
        self.robot_index = robot_index
        self.subscription

    def scan_callback(self, msg: LaserScan):
        global laser_ranges
        laser_ranges[self.robot_index] = msg.ranges

        # print(f"Updating scan data")


class GazeboEnv(Env):
    def __init__(self, goal_position=(0., 0.)):
        super(GazeboEnv, self).__init__()
        global agent_count

        # Define action space
        # action (linear_x velocity, angular_z velocity)
        # self.action_space = spaces.Box(low=np.array(
        #     [-1.0, -1.0]), high=np.array([2.0, 1.0]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array(
            [-1.5, -1.5]), high=np.array([1.5, 1.5]), dtype=np.float32)

        # observation = (lidar ranges, relative params of target, last action)
        self.output_shape = (180, 2, 2)
        # self.observation_space = spaces.Box(low=np.array(
        #     [0.0, (0, 0), (-1., -1.)]), high=np.array([30.0, max_distance_to_goal*1.0, math.pi, 1., 1.]), shape=self.output_shape, dtype=np.float32, )

        # Flattened shape: 180 lidar ranges + 2 relative target params + 2 last action params
        # self.output_shape = (180 + 2 + 2, )
        self.observation_space = spaces.Box(low=np.concatenate((np.zeros(180), np.array([0.0, 0.0]), np.array([-1.5, -1.5]))),
                                            high=np.concatenate((np.full(180, 30.0), np.array(
                                                [max_distance_to_goal * 1.0, math.pi]), np.array([1.5, 1.5]))),
                                            dtype=np.float32)

        self.reward_range = (-200, 200)
        # self.spec.max_episode_steps = 1000
        # self.spec.name = "GazeboDDPG"

        self.node = Node('GazeboEnv')

        self.vel_pubs = {agent_index: self.node.create_publisher(
            Twist, f"/robot_{agent_index+1}/cmd_vel", 10)
            for agent_index in range(agent_count)}

        self.unpause = self.node.create_client(Empty, "/unpause_physics")
        self.pause = self.node.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.node.create_client(Empty, "/reset_world")
        self.req = Empty.Request
        self.goal_position = goal_position

        self.last_actions = {agent_index: (0.0, 0.0)
                             for agent_index in range(agent_count)}
        self.prev_distances_to_goal = [Utils.get_distance_to_goal(self.get_robot_position(
            agent_index), self.goal_position) for agent_index in range(agent_count)]

    def step(self, action: Any, agent_index: int) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        global odom_data
        global laser_ranges

        linear_x = float(action[0])
        angular_z = float(action[1])

        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z

        print(
            f"Publishing velocities {action}")

        self.vel_pubs[agent_index].publish(msg)
        self.last_actions[agent_index] = (linear_x, angular_z)

        time.sleep(0.15)

        # observation = (lidar ranges, relative params of target, last action)
        observation = self.get_obs(agent_index)

        terminated = self.check_done(agent_index)
        reward = self.get_reward(terminated, agent_index)
        truncated = None
        info = None

        return observation, reward, terminated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        future = self.reset_proxy.call_async(Empty.Request())

        # Wait for the future to complete
        # rclpy.spin_until_future_complete(self.node, future, self.node.executor)

        def callback(msg):
            print(msg.result())

        future.add_done_callback(callback=callback)

        # if future.result() is not None:
        #     self.node.get_logger().info("Reset service call succeeded")
        # else:
        #     self.node.get_logger().error("Reset service call failed")

        observation = self.get_obs(0)
        info = None

        return observation, info

    def get_obs(self, agent_index):
        robot_position = Utils.get_position_from_odom_data(
            odom_data[agent_index])
        orientation = odom_data[agent_index].pose.pose.orientation
        robot_orientation = Utils.euler_from_quaternion(orientation)
        distance_to_goal = Utils.get_distance_to_goal(
            robot_position, self.goal_position)
        angle_to_goal = Utils.get_angle_to_goal(
            robot_position, robot_orientation, self.goal_position)

        normalized_lidar_ranges = laser_ranges[agent_index] / 30.0
        normalized_dist_to_goal = distance_to_goal / max_distance_to_goal
        normalized_angle_to_goal = angle_to_goal / np.pi

        # observation = (lidar ranges, relative params of target, last action)
        # observation = tuple(normalized_lidar_ranges) + (
        #     normalized_dist_to_goal, normalized_angle_to_goal) + tuple(self.last_actions.get(agent_index))

        observation = np.concatenate([laser_ranges[agent_index], [distance_to_goal, angle_to_goal],
                                     self.last_actions.get(agent_index)])

        # print("Observation type:", type(observation))
        # print("Observation shape:", np.shape(observation))
        # print("Observation:", observation)
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        print("did it with no problems")

        return observation

    def get_robot_position(self, agent_index):
        global odom_data
        return Utils.get_position_from_odom_data(odom_data[agent_index])

    def check_done(self, agent_index):
        if Utils.get_distance_to_goal((odom_data[agent_index].pose.pose.position.x, odom_data[agent_index].pose.pose.position.y), self.goal_position) < GOAL_REACHED_THRESHOLD:
            self.node.get_logger().info(
                f"Goal reached. Distance to goal: {Utils.get_distance_to_goal((odom_data[agent_index].pose.pose.position.x, odom_data[agent_index].pose.pose.position.y), self.goal_position)}")
            return True

        if min(laser_ranges[agent_index]) < OBSTACLE_COLLISION_THRESHOLD:
            self.node.get_logger().info(
                f"Collision detected. minRange: {min(laser_ranges[agent_index])}")
            return True

        return False

    def get_reward(self, done, agent_index: int):
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

        return total_aproach_reward

    def close(self):
        self.node.destroy_node()


def main(args=None):
    rclpy.init(args=args)

    # my_world.sdf
    goal_position = (43.618300, -0.503538)

    # small_world.sdf
    goal_position = (-8.061270, 1.007540)

    namespaces = ["robot_1", "robot_2", "robot_3"]
    namespaces = ["robot_1"]
    executor = MultiThreadedExecutor()

    checkpoints_path = data_folder + data_name + '/'

    # if not in training set epsilon as 0.0
    env = GazeboEnv(goal_position=goal_position)
    action_space_high = env.action_space.high[0]
    action_space_low = env.action_space.low[0]

    obs_space_shape = env.observation_space.shape[0]
    action_space_shape = env.action_space.shape[0]

    print(
        f"obs_space_shape: {obs_space_shape}\naction_space_shape: {action_space_shape}")

    # brain = Brain(obs_space_shape, action_space_shape,
    #               action_space_high, action_space_low)

    agent = Agent(input_dims=env.observation_space.shape,
                  env=env, n_actions=env.action_space.shape[0])

    tensorboard = Tensorboard(log_dir="./logs/")

    load_checkpoint = False
    evaluate = False

    # logging.info(
    #     "Loading weights from %s*, make sure the folder exists", checkpoints_path)
    # brain.load_weights(checkpoints_path)

    # metrics
    acc_reward = keras.metrics.Sum('reward', dtype=tf.float32)
    actions_squared = keras.metrics.Mean('actions', dtype=tf.float32)
    Q_loss = keras.metrics.Mean('Q_loss', dtype=tf.float32)
    A_loss = keras.metrics.Mean('A_loss', dtype=tf.float32)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    print("Training starting")

    for i, namespace in enumerate(namespaces):
        robot_index = i
        odom_subscriber = OdomSubscriber(namespace, robot_index)
        scan_subscriber = ScanSubscriber(namespace, robot_index)

        executor.add_node(odom_subscriber)
        executor.add_node(scan_subscriber)
        executor.add_node(env.node)

    executor_thread = threading.Thread(target=executor.spin, daemon=False)
    executor_thread.start()
    # executor_thread.run()

    # executor.spin()

    for ep in range(TOTAL_EPISODES):

        print("Actually starting now")

#
#         prev_state, info = env.reset()
#         acc_reward.reset_state()
#         actions_squared.reset_state()
#         Q_loss.reset_state()
#         A_loss.reset_state()
#         brain.noise.reset()
#
#         warm_up = 10
#         eps_greedy = 0.95
#         use_noise = False

        agent_index = 0

        observation, info = env.reset()
        done = False
        score = 0

        # max_step_size
        for step in range(2000):
            print("Acting...")

            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info = env.step(action, agent_index)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

            print(f"Episode: {ep+1} - Step: {step} - Reward: {reward}")

            train = True

#             if train:
#                 c, a = brain.learn(
#                     brain.buffer.get_batch(unbalance_p=UNBALANCE_P))
#                 Q_loss(c)
#                 A_loss(a)
#
#             # Post update for next step
#             acc_reward(reward)
#             actions_squared(np.square(cur_act/action_space_high))

            if done:
                break

        ep_reward_list.append(acc_reward.result().numpy())

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        avg_reward_list.append(avg_reward)

        # Print the average reward
        tensorboard(ep, acc_reward, actions_squared, Q_loss, A_loss)

        # Save weights
        # if train and ep % SAVE_INTERVAL == 0:
        #     brain.save_weights(checkpoints_path)

    env.close()

    # if train:
    #     brain.save_weights(checkpoints_path)

    logging.info("Training done...")

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.savefig(data_folder + "figures/reward_" + str(0) + '.png')

    rclpy.shutdown()


if __name__ == "__main__":
    main()
