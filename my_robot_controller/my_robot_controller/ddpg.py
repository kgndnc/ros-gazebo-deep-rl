import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import rclpy
import rclpy.waitable
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import time

from torch import detach

# Actor Network


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = torch.relu(self.l1(xu))
        x2 = torch.relu(self.l2(x1))
        x3 = self.l3(x2)
        return x3

# DDPG Agent


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64):
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)

        target_Q = self.critic_target(
            next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

# Replay Buffer


class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = [], [], [], [], []

        for i in ind:
            state, action, next_state, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(1 - done, copy=False))

        return (
            torch.FloatTensor(np.array(batch_states)).to(device),
            torch.FloatTensor(np.array(batch_actions)).to(device),
            torch.FloatTensor(np.array(batch_next_states)).to(device),
            torch.FloatTensor(np.array(batch_rewards)).to(device),
            torch.FloatTensor(np.array(batch_dones)).to(device)
        )

# ROS DDPG Agent


class ROS_DDPG(DDPG):
    def __init__(self, state_dim, action_dim, max_action, robot_namespace=""):
        super(ROS_DDPG, self).__init__(state_dim, action_dim, max_action)

        self.node = rclpy.create_node('ddpg_agent')
        self.lidar_data = None
        self.odom_data = None

        self.vel_pub = self.node.create_publisher(
            Twist, f'/{robot_namespace}/cmd_vel', qos_profile=10)
        self.node.create_subscription(LaserScan, f'/{robot_namespace}/scan',
                                      self.lidar_callback, qos_profile=10)
        self.node.create_subscription(Odometry, f'/{robot_namespace}/odom',
                                      self.odom_callback, qos_profile=10)

    def lidar_callback(self, data):
        print("updating lidar")
        self.lidar_data = data

    def odom_callback(self, data):
        print("updating odom")
        self.odom_data = data

    def get_state(self):
        if self.lidar_data is not None and self.odom_data is not None:
            lidar_ranges = np.array(self.lidar_data.ranges)
            odom_position = np.array(
                [self.odom_data.pose.pose.position.x, self.odom_data.pose.pose.position.y])
            odom_orientation = np.array(
                [self.odom_data.pose.pose.orientation.z, self.odom_data.pose.pose.orientation.w])
            state = np.concatenate(
                (lidar_ranges, odom_position, odom_orientation))
            return state
        else:
            return None

    def publish_action(self, action):
        cmd = Twist()
        cmd.linear.x = action[0]
        cmd.angular.z = action[1]
        print(f"publishing {(action,)}")
        self.vel_pub.publish(cmd)

    def compute_reward(self, state):
        r_collision = -100
        r_arrived = +200
        # TODO: calculate dist. to target and define a reward for it
        if min(self.lidar_data.ranges < 0.35):
            return r_collision

        return 1

    def check_done(self, state):
        # Define your own done condition here
        return False


rclpy.init()

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = 362  # 360 lidar ranges + 2 odom position + 2 odom orientation
state_dim = 181 + 1 + 1
action_dim = 2  # linear velocity and angular velocity
max_action = 1.0

# Training
replay_buffer = ReplayBuffer()
ros_ddpg_agent = ROS_DDPG(state_dim, action_dim,
                          max_action, robot_namespace="robot_1")

episodes = 100
batch_size = 64
reset_node = rclpy.create_node("reset_node")
reset_simulation = reset_node.create_client(
    Empty, "reset_world")


for episode in range(episodes):
    # while reset_simulation.wait_for_service(timeout_sec=1):
    #     reset_node.get_logger().info("Waiting for reset service...")

    reset_simulation.call_async(Empty.Request())

    print("hello")

    state = ros_ddpg_agent.get_state()
    while state is None:
        state = ros_ddpg_agent.get_state()

    episode_reward = 0

    for step in range(200):
        print("inside")

        action = ros_ddpg_agent.select_action(np.array(state))
        ros_ddpg_agent.publish_action(action)
        # Sleep for a short duration to allow Gazebo to update
        time.sleep(0.1)

        next_state = ros_ddpg_agent.get_state()
        while next_state is None:
            next_state = ros_ddpg_agent.get_state()

        # Assuming a reward function and done signal are provided by the environment
        reward = ros_ddpg_agent.compute_reward(next_state)
        done = ros_ddpg_agent.check_done(next_state)

        replay_buffer.add((state, action, next_state, reward, done))
        state = next_state
        episode_reward += reward

        if len(replay_buffer.storage) > batch_size:
            ros_ddpg_agent.train(replay_buffer, batch_size)

        if done:
            break

    print(f"Episode: {episode+1}, Reward: {episode_reward}")

rclpy.shutdown()
