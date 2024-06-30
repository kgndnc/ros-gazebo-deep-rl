import threading
import rclpy
from rclpy.executors import MultiThreadedExecutor
from subscribers import OdomSubscriber, ScanSubscriber

from replay_buffer import ReplayBuffer
from GazeboEnv import GazeboEnvMultiAgent

from config import *


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter


# TODO:
#   - dynamic target spawn (place the goal close first then far)
#   - change optimizer Adam to RAdam


# td3 code
# ===============================================

device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # cuda or cpu


def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        env.node.get_logger().info(f"evaluating episode {_}")
        count = 0
        state_n = env.reset()
        done_n = [False]
        while not any(done_n) and count < 501:
            # action = network.get_action(np.array(state))
            # env.node.get_logger().info(f"action : {action}")
            # a_in = [(action[0] + 1) / 2, action[1]]
            # state, reward, done, _ = env.step(a_in)
            # avg_reward += reward
            # count += 1

            iter_reward = 0
            action_n = []
            for i in range(AGENT_COUNT):
                action = network.get_action(np.array(state_n[i]))
                env.node.get_logger().info(f"action : {action}")
                action_n.append(action)

            a_in_n = []
            for i, action in enumerate(action_n):
                a_in = [(action[0] + 1) / 2, action[1]]
                a_in_n.append(a_in)

            state_n, reward_n, done_n, _ = env.step(a_in_n)
            for r in reward_n:
                iter_reward += r
                avg_reward += r
            count += 1

            # if iter_reward < -90 * AGENT_COUNT:
            if iter_reward < -90:
                col += 1

    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    env.node.get_logger().info("..............................................")
    env.node.get_logger().info(
        "Average Reward over %i Evaluation Episodes, Epoch %i: avg_reward %f, avg_col %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    env.node.get_logger().info("..............................................")
    return avg_reward


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2

# td3 network


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter(
            log_dir="./runs")
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0

        print(f"train function iteration count: {iterations}")
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(
                0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (
                next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + \
                F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        # Write new values for tensorboard
        env.node.get_logger().info(f"writing new results for a tensorboard")
        env.node.get_logger().info(
            f"loss, Av.Q, Max.Q, iterations : {av_loss / iterations}, {av_Q / iterations}, {max_Q}, {self.iter_count}")
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

        # TODO (optional): add hyperparameters to tensorboard
        # self.writer.add_hparams()

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" %
                   (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" %
                   (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )


# td3 code end
# ===============================================


if __name__ == "__main__":
    rclpy.init()

    seed = 0  # Random seed number
    eval_freq = 5e3  # After how many steps to perform the evaluation
    max_ep = 500  # maximum number of steps per episode
    eval_ep = 10  # number of episodes for evaluation
    max_timesteps = 5e6  # Maximum number of steps to perform
    # Initial exploration noise starting value in range [expl_min ... 1]
    expl_noise = 1
    expl_decay_steps = (
        500000  # Number of steps over which the initial exploration noise will decay over
    )
    # Exploration noise after the decay in range [0...expl_noise]
    expl_min = 0.1
    batch_size = 40  # Size of the mini-batch
    # Discount factor to calculate the discounted future reward (should be close to 1)
    discount = 0.99999
    tau = 0.005  # Soft target update variable (should be close to 0)
    policy_noise = 0.2  # Added noise for exploration
    noise_clip = 0.5  # Maximum clamping values of the noise
    policy_freq = 2  # Frequency of Actor network updates
    buffer_size = 1e6  # Maximum size of the buffer
    file_name = "td3_policy"  # name of the file to store the policy
    save_model = True  # Whether to save the model or not
    load_model = True  # Whether to load a stored model
    random_near_obstacle = True  # To take random actions near obstacles or not

    # Create the network storage folders
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_model and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Create the training environment
    # laser sample count
    environment_dim = LIDAR_SAMPLE_SIZE
    robot_dim = 4

    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim + robot_dim
    action_dim = 2
    max_action = 1

    # Create the network
    network = TD3(state_dim, action_dim, max_action)
    # Create a replay buffer
    replay_buffer = ReplayBuffer(buffer_size, seed)
    if load_model:
        try:
            print("Will load existing model.")
            network.load(file_name, "./pytorch_models")
        except:
            print(
                "Could not load the stored model parameters, initializing training with random parameters")

    # Create evaluation data store
    evaluations = []

    timestep = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    epoch = 1
    episode_timesteps = 0
    episode_reward = 0

    count_rand_actions = 0
    random_action = []

    goal_position = (-8.061270, 1.007540)
    # namespaces = ["robot_1", "robot_2", "robot_3"]
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

    env = GazeboEnvMultiAgent(odom_subscribers=odom_subscribers,
                              scan_subscribers=scan_subscribers, goal_position=goal_position)
    executor.add_node(env.node)

    executor_thread = threading.Thread(target=executor.spin, daemon=False)
    executor_thread.start()

    # get observation shapes []
    obs_shape_n = [
        env.observation_space.shape for i in range(env.agent_count)]

    action_space_n = [env.action_space for i in range(env.agent_count)]

    # # benchmark info (rewards, agent info etc.)
    # episode_rewards = [0.0]  # sum of rewards for all agents
    # agent_rewards = [[0.0]
    #                  for _ in range(env.n)]  # individual agent reward
    # final_ep_rewards = []  # sum of rewards for training curve
    # final_ep_ag_rewards = []  # agent rewards for training curve
    # agent_info = [[[]]]  # placeholder for benchmarking info
    # # saver = tf.train.Checkpoint()
    # episode_step = 0
    # train_step = 0
    # t_start = time.time()

    # reset environment
    prev_observation_n = env.reset()
    just_reset = True

    print('Starting iterations...')

    # training loop:
    while timestep < max_timesteps:
        if done:
            env.node.get_logger().info(
                f"Done. Episode num: {episode_num} - Total episode rewards: {episode_reward} ")
            if timestep != 0:
                env.node.get_logger().info(f"Training network")
                network.train(replay_buffer,
                              episode_timesteps,
                              batch_size,
                              discount,
                              tau,
                              policy_noise,
                              noise_clip,
                              policy_freq,
                              )
                if timesteps_since_eval >= eval_freq:
                    env.node.get_logger().info("Validating")
                    timesteps_since_eval %= eval_freq
                    evaluations.append(
                        evaluate(network=network, epoch=epoch,
                                 eval_episodes=eval_ep)
                    )
                    if save_model:
                        network.save(
                            file_name, directory="./pytorch_models")
                        np.save("./results/%s" % (file_name), evaluations)
                    epoch += 1

                prev_observation_n = env.reset()
                done = False
                just_reset = True

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

             # add some exploration noise

        if expl_noise > expl_min:
            expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

        # get actions
        action_n = []
        for i in range(AGENT_COUNT):
            action = network.get_action(np.array(prev_observation_n[i]))
            action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                -max_action, max_action
            )
            action_n.append(action)

        # random near obstacle (add later if you want)

        # scale actions to fit proper ranges
        a_in_n = []
        for i, action in enumerate(action_n):
            a_in = [(action[0] + 1) / 2, action[1]]
            a_in_n.append(a_in)

        observation, reward, done_n, info = env.step(a_in_n)
        if just_reset:
            reward = [0.0 for _ in range(AGENT_COUNT)]
            just_reset = False

        done_bool = 0 if episode_timesteps + \
            1 == max_ep else int(any(done_n))
        done = 1 if episode_timesteps + 1 == max_ep else int(any(done_n))

        for i, r in enumerate(reward):
            episode_reward += r
            print(f"Reward for agent_{i}: {r}")

        # save experience to buffer
        for i in range(AGENT_COUNT):
            replay_buffer.add(
                prev_observation_n[i], action_n[i], reward[i], done_bool, observation[i])

        prev_observation_n = observation
        episode_timesteps += 1
        timestep += 1
        timesteps_since_eval += 1

    rclpy.shutdown()
