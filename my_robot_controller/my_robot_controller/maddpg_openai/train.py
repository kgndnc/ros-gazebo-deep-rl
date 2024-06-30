import threading
import rclpy
from rclpy.executors import MultiThreadedExecutor
from subscribers import OdomSubscriber, ScanSubscriber
from GazeboEnv import GazeboEnv
from maddpg import MADDPGAgentTrainer
import tf_util as U
from config import MAX_STEPS


import argparse
import numpy as np
import tensorflow
import time
import pickle

import keras._tf_keras.keras.layers as layers

tf = tensorflow.compat.v1

tf.disable_eager_execution()


variable_scope = tf.compat.v1.variable_scope


def _mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(
            out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(
            out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(
            out, num_outputs=num_outputs, activation_fn=None)
        return out


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions

    with variable_scope(scope, reuse=reuse):
        out = input

        out = layers.Dense(num_units, activation=tf.nn.relu)(out)
        out = layers.Dense(num_units, activation=tf.nn.relu)(out)
        out = layers.Dense(num_outputs, activation=None)(out)

        return out


if __name__ == "__main__":
    rclpy.init()
    goal_position = (-8.061270, 1.007540)
    namespaces = ["robot_1", "robot_2", "robot_3"]

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

    # Create environment
    # env = GazeboEnv(odom_subscribers, scan_subscribers,
    #                 goal_position=goal_position)

    with U.single_threaded_session():

        env = GazeboEnv(odom_subscribers=odom_subscribers,
                        scan_subscribers=scan_subscribers, goal_position=goal_position)
        executor.add_node(env.node)

        executor_thread = threading.Thread(target=executor.spin, daemon=False)
        executor_thread.start()

        # get observation shapes []
        obs_shape_n = [
            env.observation_space.shape for i in range(env.agent_count)]

        model = mlp_model
        # get trainers []

        # action_space_n is a list like obs_shape_n

        action_space_n = [env.action_space for i in range(env.agent_count)]

        # U.init()
        U.initialize()

        trainers = [MADDPGAgentTrainer("agent_%d" % i, model, obs_shape_n, action_space_n, i, None, False)
                    for i in range(env.agent_count)]

        # Load previous results, if necessary

        # benchmark info (rewards, agent info etc.)
        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0]
                         for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Checkpoint()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        # reset environment
        obs_n = env.reset()

        print('Starting iterations...')

        # train loop:
        while True:
            #   get action from agent.action()
            action_n = [agent.action(obs)
                        for agent, obs in zip(trainers, obs_n)]

            # env.step() (get obs, reward...)
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            # check done
            done = all(done_n)
            terminal = (episode_step >= MAX_STEPS)
            # collect experience agent.experience()
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i],
                                 rew_n[i], new_obs_n[i], done_n[i], terminal)

            # transition state
            obs_n = new_obs_n

            # add episode rewards, agent rewards etc.

            # reset environment if done or terminal
            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)
