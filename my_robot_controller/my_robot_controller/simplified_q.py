import numpy as np
import random
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
# from squaternion import Quaternion
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty

# Parameters
state_size = (20, 20)  # Example discretized state space
action_size = 4  # Example number of actions
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 1.0
exploration_decay = 0.995
min_exploration_rate = 0.01
max_steps_per_episode = 200
episodes = 1000

# Actions (forward, backward, left, right)
actions = [(-0.5, 0), (0.5, 0), (0, 0.5), (0, -0.5)]

# Q-Table
q_table = np.zeros(state_size + (action_size,))


def discretize_state(odom, laser_data):
    # Discretize odometry and laser scan data
    x = int(min(max(odom.pose.pose.position.x + 10, 0), 19))
    y = int(min(max(odom.pose.pose.position.y + 10, 0), 19))
    return (x, y)


def get_reward(state, done):
    # Reward function
    if done:
        return -100
    elif state == (10, 10):  # Example goal position
        return 100
    else:
        return -1


class QLearningAgent(Node):
    def __init__(self):
        super().__init__('q_learning_agent')
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.unpause = self.create_client(Empty, '/unpause_physics')
        self.pause = self.create_client(Empty, '/pause_physics')
        self.reset_proxy = self.create_client(Empty, '/reset_world')
        self.odom = None
        self.laser_data = None
        self.state = None

    def odom_callback(self, msg):
        self.odom = msg

    def laser_callback(self, msg):
        self.laser_data = msg.ranges

    def choose_action(self, state):
        if random.uniform(0, 1) < exploration_rate:
            return random.choice(range(action_size))
        else:
            return np.argmax(q_table[state])

    def take_action(self, action):
        twist = Twist()
        twist.linear.x, twist.angular.z = actions[action]
        self.vel_pub.publish(twist)

    def step(self, action):
        self.take_action(action)
        time.sleep(0.1)
        next_state = discretize_state(self.odom, self.laser_data)
        done = self.check_done()
        reward = get_reward(next_state, done)
        return next_state, reward, done

    def check_done(self):
        # Check for collisions or reaching goal
        if min(self.laser_data) < 0.2:
            return True
        if (self.odom.pose.pose.position.x, self.odom.pose.pose.position.y) == (10, 10):
            return True
        return False

    def reset(self):
        self.reset_proxy.call_async(Empty.Request())
        time.sleep(0.5)
        self.odom = None
        self.laser_data = None
        while self.odom is None or self.laser_data is None:
            rclpy.spin_once(self)
        self.state = discretize_state(self.odom, self.laser_data)
        return self.state


def main():
    rclpy.init()
    agent = QLearningAgent()
    global exploration_rate

    for episode in range(episodes):
        state = agent.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done = agent.step(action)
            q_table[state][action] = q_table[state][action] + learning_rate * (
                reward + discount_factor *
                np.max(q_table[next_state]) - q_table[state][action]
            )
            state = next_state
            total_reward += reward
            if done:
                break

        exploration_rate = max(min_exploration_rate,
                               exploration_rate * exploration_decay)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    np.save("q_table.npy", q_table)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
