import threading
import time
from typing import Any, List, SupportsFloat

import keras.api
import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
import tensorflow as tf
from keras import layers
import keras
import os

import math

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from collections import namedtuple
import sys

sys.path.append("./environment")  # nopep8

from environment.config import *
from environment.subscribers import OdomSubscriber, ScanSubscriber
from environment.GazeboEnv import GazeboEnvMultiAgent
from environment.Utils import Utils


Observation = namedtuple('Observation', ['lidar', 'state_params'])


"""
TODO:
    - update network structure
    - MaxPooling olayı
    - find out if inputs need to be normalized [0, 1] or [-1, 1] (StatQuest video)
    - change model
"""


GOAL_REACHED_THRESHOLD = 1.0
OBSTACLE_COLLISION_THRESHOLD = 0.7
LIDAR_SAMPLE_SIZE = 180
SAVE_INTERVAL = 5


os.environ["KERAS_BACKEND"] = "tensorflow"

# small_world.sdf
bounds = ((-10, 10), (-10, 14))
x_grid_size = bounds[0][1] - bounds[0][0]  # Define the grid size
y_grid_size = bounds[1][1] - bounds[1][0]  # Define the grid size
# hypotenuse of the environment - radius of the robot
max_distance_to_goal = math.floor(
    math.sqrt(x_grid_size**2 + y_grid_size**2) - 0.6)
max_distance_to_goal *= 1.0
# global variables for sensor data
agent_count = 3


class GazeboEnv(GazeboEnvMultiAgent):
    def __init__(self, odom_subscribers: List[OdomSubscriber], scan_subscribers: List[ScanSubscriber], goal_position=...):
        super().__init__(odom_subscribers, scan_subscribers, goal_position)

    def get_obs(self):
        # return observations list
        # observation = [lidar ranges, distance to goal, angle to goal, last action]
        observations = []
        max_lidar_range = MAX_LIDAR_RANGE

        for i in range(self.agent_count):
            robot_position = Utils.get_position_from_odom_data(
                self.odom_subscribers[i].odom_data)
            orientation = self.odom_subscribers[i].odom_data.pose.pose.orientation
            robot_orientation = Utils.euler_from_quaternion(orientation)
            distance_to_goal = Utils.get_distance_to_goal(
                robot_position, self.goal_position)
            angle_to_goal = Utils.get_angle_to_goal(
                robot_position, robot_orientation, self.goal_position)

            # reduce 180 samples to 20
            # reduced_lidar_ranges = Utils.reduce_lidar_samples(
            #     self.scan_subscribers[i].laser_ranges, 20)

            normalized_lidar_ranges = self.scan_subscribers[i].laser_ranges / \
                max_lidar_range

            normalized_lidar_ranges = np.clip(
                normalized_lidar_ranges, 0.0, 1.0)
            normalized_dist_to_goal = distance_to_goal / max_distance_to_goal
            normalized_angle_to_goal = angle_to_goal / np.pi

            state_parameter_set = np.concatenate(
                [[normalized_dist_to_goal, normalized_angle_to_goal],
                    [self.last_actions.get(i)[0],
                     self.last_actions.get(i)[1]]]
            )

            # observation = np.concatenate(
            #     [normalized_lidar_ranges, state_parameter_set])

            observation = Observation(
                lidar=normalized_lidar_ranges, state_params=state_parameter_set)

            observations.append(observation)

        # return observations list
        # observation = [lidar ranges, distance to goal, angle to goal, last action]

        # print(f"obs: {observations}")

        return observations

    def reset(self):

        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("/reset_world service not available, waiting")

        try:
            self.reset_proxy.call_async(Empty.Request())
        except:
            self.node.get_logger().error("/reset_world service call failed!")

        self.change_goal_position()

        time.sleep(0.1)

        observations = self.get_obs()

        return observations


# small_world.sdf
goal_position = (-8.061270, 1.007540)

# Specify the `render_mode` parameter to show the attempts of the agent in a pop up window.
# env = gym.make("Pendulum-v1", render_mode="human")

rclpy.init()

goal_position = (-8.061270, 1.007540)

namespaces = [f"robot_{i+1}" for i in range(AGENT_COUNT)]

# set up subscribers and environment
executor = MultiThreadedExecutor()
odom_subscribers = []
scan_subscribers = []
for i, namespace in enumerate(namespaces):
    robot_index = i
    odom_subscriber = OdomSubscriber(namespace, robot_index)
    scan_subscriber = ScanSubscriber(namespace, robot_index)
    odom_subscribers.append(odom_subscriber)
    scan_subscribers.append(scan_subscriber)
    executor.add_node(odom_subscriber)
    executor.add_node(scan_subscriber)

env = GazeboEnv(odom_subscribers=odom_subscribers,
                scan_subscribers=scan_subscribers, goal_position=goal_position)


num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) *
            np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        # self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        # self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        # self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        # self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

        # new imp.
        # self.lidar_buffer = np.zeros(
        #     (self.buffer_capacity, LIDAR_SAMPLE_SIZE, 1))
        self.lidar_buffer = np.zeros(
            (self.buffer_capacity, LIDAR_SAMPLE_SIZE))
        self.state_params_buffer = np.zeros((self.buffer_capacity, 4))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        # self.next_lidar_buffer = np.zeros(
        #     (self.buffer_capacity, LIDAR_SAMPLE_SIZE, 1))
        self.next_lidar_buffer = np.zeros(
            (self.buffer_capacity, LIDAR_SAMPLE_SIZE))
        self.next_state_params_buffer = np.zeros((self.buffer_capacity, 4))

    # Takes (s,a,r,s') observation tuple as input

    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        # self.state_buffer[index] = obs_tuple[0]
        # self.action_buffer[index] = obs_tuple[1]
        # self.reward_buffer[index] = obs_tuple[2]
        # self.next_state_buffer[index] = obs_tuple[3]

        # new imp.
        self.lidar_buffer[index] = obs_tuple[0].lidar
        self.state_params_buffer[index] = obs_tuple[0].state_params
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_lidar_buffer[index] = obs_tuple[3].lidar
        self.next_state_params_buffer[index] = obs_tuple[3].state_params

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    # def update(
    #     self,
    #     state_batch,
    #     action_batch,
    #     reward_batch,
    #     next_state_batch,
    # ):
    @ tf.function
    def update(
        self,
        lidar_batch, state_params_batch,
        action_batch,
        reward_batch,
        next_lidar_batch, next_state_params_batch
    ):

        # state batch = [ lidar, state param. set ]
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            # target_actions = target_actor(next_state_batch, training=True)
            # y = reward_batch + gamma * target_critic(
            #     [next_state_batch, target_actions], training=True
            # )
            # critic_value = critic_model(
            #     [state_batch, action_batch], training=True)
            # critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))
            target_actions = target_actor(
                [next_lidar_batch, next_state_params_batch], training=True)
            target_actions_concat = tf.concat(target_actions, axis=-1)
            # print([next_lidar_batch, next_state_params_batch, target_actions_concat])
            y = reward_batch + gamma * target_critic(
                [next_lidar_batch, next_state_params_batch, target_actions_concat],
                training=True)
            critic_value = critic_model(
                [lidar_batch, state_params_batch, action_batch], training=True)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

        critic_grad = tape.gradient(
            critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(
                [lidar_batch, state_params_batch], training=True)
            actions_concat = tf.concat(actions, axis=-1)

            critic_value = critic_model(
                [lidar_batch, state_params_batch, actions_concat], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -keras.ops.mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

        # print(
        #     f"Max of actor_grad[0]: {tf.reduce_max(tf.get_static_value(actor_grad)[-1]  ):.20f}")
        # print(
        #     f"Min of actor_grad[0]: {tf.reduce_min(tf.get_static_value(actor_grad)[-1]):.20f}")
        # print(
        #     f"Mean of actor_grad[0]: {tf.reduce_mean(tf.get_static_value(actor_grad)[-1]):.20f}")
        # print("Max of actor_grad[0]: %.4f" % tf.reduce_max(actor_grad[0]))
        # print("Min of actor_grad[0]: %.4f" % tf.reduce_min(actor_grad[0]))

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
#         state_batch = keras.ops.convert_to_tensor(
#             self.state_buffer[batch_indices])
#         action_batch = keras.ops.convert_to_tensor(
#             self.action_buffer[batch_indices])
#         reward_batch = keras.ops.convert_to_tensor(
#             self.reward_buffer[batch_indices])
#         reward_batch = keras.ops.cast(reward_batch, dtype="float32")
#         next_state_batch = keras.ops.convert_to_tensor(
#             self.next_state_buffer[batch_indices]
#         )
#
#         self.update(state_batch, action_batch, reward_batch, next_state_batch)

        # new imp.
        lidar_batch = keras.ops.convert_to_tensor(
            self.lidar_buffer[batch_indices])
        state_params_batch = keras.ops.convert_to_tensor(
            self.state_params_buffer[batch_indices])
        action_batch = keras.ops.convert_to_tensor(
            self.action_buffer[batch_indices])
        reward_batch = keras.ops.convert_to_tensor(
            self.reward_buffer[batch_indices])
        reward_batch = keras.ops.cast(reward_batch, dtype="float32")
        next_lidar_batch = keras.ops.convert_to_tensor(
            self.next_lidar_buffer[batch_indices])
        next_state_params_batch = keras.ops.convert_to_tensor(
            self.next_state_params_buffer[batch_indices])

        self.update(lidar_batch, state_params_batch, action_batch,
                    reward_batch, next_lidar_batch, next_state_params_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(target, original, tau):
    target_weights = target.get_weights()
    original_weights = original.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = original_weights[i] * \
            tau + target_weights[i] * (1 - tau)

    target.set_weights(target_weights)


"""
Inputs:
    - Lidar Data of 180 samples
    - State paramater set -> (target_angle, target_distance, last_action)

Output:
    - action (linear_velocity, angular_velocity)

"""

lidar_input_shape = (LIDAR_SAMPLE_SIZE, 1)


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

    # Lidar data feature extraction
    lidar_input = layers.Input(shape=lidar_input_shape)
    x = layers.Conv1D(64, (7,), strides=3, padding='same')(lidar_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.Conv1D(64, (3,), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, (3,), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling1D(pool_size=3)(x)
    x = layers.Flatten()(x)
    x = keras.Model(inputs=lidar_input, outputs=x)

    print(f"Lidar feature extraction output shape: {x.output_shape}")
    print(f"Lidar feature extraction output: {x.output}")

    # There is more input (target_angle, target_distance, last_action)
    # Define the dimensions of the additional state inputs
    state_dim = 4  # For θ, ρ, v_t-1, ω_t-1
    state_input = layers.Input(shape=(state_dim,))

    # Concatenate lidar features with additional state input
    concat = layers.Concatenate(axis=-1)([x.output, state_input])

    y = layers.Dense(512, activation='relu',
                     kernel_initializer='he_uniform')(concat)
    y = layers.Dense(256, activation='relu',
                     kernel_initializer='he_uniform')(y)
    y = layers.Dense(128, activation='relu',
                     kernel_initializer='he_uniform')(y)

    linear_output = layers.Dense(
        1, activation='tanh', kernel_initializer="glorot_uniform")(y)
    # angular_output = layers.Dense(
    #     1, activation='sigmoid', kernel_initializer=last_init)(y)
    angular_output = layers.Dense(
        1, activation='tanh', kernel_initializer="glorot_uniform")(y)

    # outputs = outputs * upper_bound

    # model = keras.Model([lidar_input, state_input], [
    #                     linear_output, angular_output])

    model = keras.Model(inputs=[x.input, state_input], outputs=[
        linear_output, angular_output])

    return model


def get_critic():
    # Input'lardaki shape'leri kontrol et

    lidar_input = layers.Input(shape=lidar_input_shape)
    state_dim = 4  # For θ, ρ, v_t-1, ω_t-1
    state_input = layers.Input(shape=(state_dim,))
    action_input = layers.Input(shape=(num_actions,))

    # Lidar data feature extraction
    x = layers.Conv1D(64, (7,), strides=3, padding='same')(lidar_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.Conv1D(64, (3,), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, (3,), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling1D(pool_size=3)(x)
    x = layers.Flatten()(x)
    x = keras.Model(inputs=lidar_input, outputs=x)

    # Concatenate lidar features with action input and additional state input
    concat = layers.Concatenate(axis=-1)([x.output, state_input, action_input])

    # Fully-connected layers (FC)
    y = layers.Dense(512, activation='relu',
                     kernel_initializer='he_uniform')(concat)
    y = layers.Dense(256, activation='relu',
                     kernel_initializer='he_uniform')(y)
    y = layers.Dense(128, activation='relu',
                     kernel_initializer='he_uniform')(y)

    # Output layer for Q-value
    q_value_output = layers.Dense(1, activation='linear')(y)

    model = keras.Model(inputs=[x.input, state_input, action_input],
                        outputs=[q_value_output])

    return model

    # @old-version
    # State as input
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through separate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object):
    # print("state")
    # print(state)

    sampled_actions = keras.ops.squeeze(actor_model(state))
    noise = noise_object()

    # print("action without noise: ", sampled_actions.numpy())

    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    # return [np.squeeze(legal_action)]
    return legal_action


load_models = False
save_models = True
train = True


# main function
if __name__ == "__main__":
    # rclpy.init()
    # small_world.sdf
    goal_position = (-8.061270, 1.007540)

    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(
        1), std_deviation=float(std_dev) * np.ones(1))

    actor_model = get_actor()
    critic_model = get_critic()

    target_actor = get_actor()
    target_critic = get_critic()

    checkpoint_dir = os.path.join('.', 'updated_ddpg')

    if load_models:
        print("...Loading weights...")
        actor_model.load_weights(os.path.join(checkpoint_dir, 'an.weights.h5'))
        critic_model.load_weights(os.path.join(
            checkpoint_dir, 'cn.weights.h5'))
        target_actor.load_weights(os.path.join(
            checkpoint_dir, 'at.weights.h5'))
        target_critic.load_weights(
            os.path.join(checkpoint_dir, 'ct.weights.h5'))
    else:
        # Making the weights equal initially
        target_actor.set_weights(actor_model.get_weights())
        target_critic.set_weights(critic_model.get_weights())

    # Learning rate for actor-critic models
    critic_lr = 0.002
    actor_lr = 0.001

    # modified according to paper
    critic_lr = 0.0001
    actor_lr = 0.001

    critic_optimizer = keras.optimizers.Adam(critic_lr)
    actor_optimizer = keras.optimizers.Adam(actor_lr)

    # actor_model.compile(
    #     optimizer=actor_optimizer,
    #     loss="mse",
    #     run_eagerly=True,
    # )
    # critic_model.compile(
    #     optimizer=critic_optimizer,
    #     loss="mse",
    #     run_eagerly=True,
    # )

    # try this
    critic_optimizer = keras.optimizers.Adam(critic_lr,  clipvalue=.5)
    actor_optimizer = keras.optimizers.Adam(actor_lr, clipvalue=.5)

    total_episodes = 10_000
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.005

    # hyperparameters
    # for single agent
    buffer_size = 100_000
    batch_size = 128
    max_steps = 800

    buffer = Buffer(buffer_capacity=buffer_size, batch_size=batch_size)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    executor_thread = threading.Thread(target=executor.spin, daemon=False)
    executor_thread.start()

    for ep in range(total_episodes):
        agent_index = 0
        prev_state_n = env.reset()
        episodic_reward = 0
        step = 1

        print("prev_state_n: ", prev_state_n)

        while step < max_steps:
            # tf_prev_state = keras.ops.expand_dims(
            #     keras.ops.convert_to_tensor(prev_state), 0
            # )

            action_n = []

            for prev_state in prev_state_n:
                lidar_obs = prev_state.lidar
                state_parameter_set_obs = prev_state.state_params

                tf_lidar_obs = keras.ops.expand_dims(
                    keras.ops.convert_to_tensor(lidar_obs), 0
                )
                tf_state_param_set = keras.ops.expand_dims(
                    keras.ops.convert_to_tensor(state_parameter_set_obs), 0
                )

                action = policy([tf_lidar_obs, tf_state_param_set], ou_noise)
                action_n .append(action)

            # Receive state and reward from environment.

            state_n, reward_n, done_n, _ = env.step(action_n)

            for i, r in enumerate(reward_n):
                print(f"Reward for agent_{i}: {r}")

            for i in range(len(state_n)):
                buffer.record(
                    (prev_state_n[i],  action_n[i], reward_n[i], state_n[i])
                )
                episodic_reward += reward_n[i]

            buffer.learn()

            update_target(target_actor, actor_model, tau)
            update_target(target_critic, critic_model, tau)

            # End this episode when `done` or `truncated` is True
            if any(done_n):
                break

            prev_state_n = state_n
            # print(f"Episode: {ep+1} - Step: {step} - Reward: {reward}")
            step += 1

        ep_reward_list.append(episodic_reward)

        # Save weights
        if train and ep % SAVE_INTERVAL == 0:
            print("...Saving weights...")
            actor_model.save_weights(os.path.join(
                checkpoint_dir, 'an.weights.h5'))
            critic_model.save_weights(os.path.join(
                checkpoint_dir, 'cn.weights.h5'))
            target_actor.save_weights(os.path.join(
                checkpoint_dir, 'at.weights.h5'))
            target_critic.save_weights(
                os.path.join(checkpoint_dir, 'ct.weights.h5'))

            # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    env.close()

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic Reward")
    plt.show()
    rclpy.shutdown()
