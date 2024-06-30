from gymnasium import Env, spaces
import numpy as np
import time
from typing import Any, List, SupportsFloat
from rclpy.node import Node
from gymnasium import Env, spaces
from Utils import Utils
import math
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist

from config import GOAL_REACHED_THRESHOLD, OBSTACLE_COLLISION_THRESHOLD, LIDAR_SAMPLE_SIZE, SAVE_INTERVAL, AGENT_COUNT, MAX_LIDAR_RANGE


# small_world.sdf
bounds = ((-10, 10), (-10, 14))
x_grid_size = bounds[0][1] - bounds[0][0]  # Define the grid size
y_grid_size = bounds[1][1] - bounds[1][0]  # Define the grid size
# hypotenuse of the environment - radius of the robot
max_distance_to_goal = math.floor(
    math.sqrt(x_grid_size**2 + y_grid_size**2) - 0.6)
max_distance_to_goal = math.floor(
    math.sqrt(x_grid_size**2 + y_grid_size**2))
max_distance_to_goal *= 1.0

agent_count = AGENT_COUNT


class GazeboEnv(Env):
    from subscribers import OdomSubscriber, ScanSubscriber

    def __init__(self, odom_subscribers: List[OdomSubscriber], scan_subscribers: List[ScanSubscriber], goal_position=(0., 0.)):
        super(GazeboEnv, self).__init__()

        self.odom_subscribers = odom_subscribers
        self.scan_subscribers = scan_subscribers

        # Define action space
        # action (linear_x velocity, angular_z velocity)
        self.action_space = spaces.Box(low=np.array(
            [-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # observation = (lidar ranges, relative params of target, last action)
        self.output_shape = (184,)
        self.observation_space = spaces.Box(low=np.concatenate((np.zeros(LIDAR_SAMPLE_SIZE), np.array([0.0, -1.0]), np.array([-1.0, -1.0]))),
                                            high=np.concatenate((np.full(LIDAR_SAMPLE_SIZE, 1.0), np.array(
                                                [1.0, 1.0]), np.array([1.0, 1.0]))),
                                            dtype=np.float32)

        self.agent_count = AGENT_COUNT

        self.reward_range = (-200, 200)

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

    def step(self, action_n: List):

        cmds = []

        for i in range(self.agent_count):
            linear_x = float(action_n[i][0])
            angular_z = float(action_n[i][1])
            msg = Twist()
            msg.linear.x = linear_x
            msg.angular.z = angular_z
            cmds.append(msg)

        print(f"Publishing velocities {action_n}")

        for i in range(self.agent_count):
            self.vel_pubs[i].publish(msg)
            self.last_actions[i] = (linear_x, angular_z)

        time.sleep(0.15)

        # observation = (lidar ranges, relative params of target, last action)
        observation = self.get_obs()

        terminated = self.check_done()
        reward = self.get_reward(terminated)
        info = [None for _ in range(self.agent_count)]

        return observation, reward, terminated, info

    def reset(self):
        future = self.reset_proxy.call_async(Empty.Request())

        def callback(msg):
            print(msg.result())

        future.add_done_callback(callback=callback)

        time.sleep(0.1)

        # if future.result() is not None:
        #     self.node.get_logger().info("Reset service call succeeded")
        # else:
        #     self.node.get_logger().error("Reset service call failed")

        observations = self.get_obs()

        return observations

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

            observation = np.concatenate(
                [normalized_lidar_ranges, state_parameter_set])

            observations.append(observation)

        # return observations list
        # observation = [lidar ranges, distance to goal, angle to goal, last action]

        # print(f"obs: {observations}")

        return observations

    def get_obs_id(self, agent_index):
        robot_position = Utils.get_position_from_odom_data(
            self.odom_subscribers[agent_index].odom_data)
        orientation = self.odom_subscribers[agent_index].odom_data.pose.pose.orientation
        robot_orientation = Utils.euler_from_quaternion(orientation)
        distance_to_goal = Utils.get_distance_to_goal(
            robot_position, self.goal_position)
        angle_to_goal = Utils.get_angle_to_goal(
            robot_position, robot_orientation, self.goal_position)

        normalized_lidar_ranges = self.scan_subscribers[agent_index].laser_ranges / \
            MAX_LIDAR_RANGE
        normalized_lidar_ranges = np.clip(
            normalized_lidar_ranges, 0.0, 1.0)
        normalized_dist_to_goal = distance_to_goal / max_distance_to_goal
        normalized_angle_to_goal = angle_to_goal / np.pi

        state_parameter_set = np.concatenate(
            [[normalized_dist_to_goal, normalized_angle_to_goal],
                [self.last_actions.get(agent_index)[0],
                 self.last_actions.get(agent_index)[1]]]
        )

        observation = np.concatenate(
            [normalized_lidar_ranges, state_parameter_set])

        # print(f"obs: {observation}")

        return observation

    def get_robot_position(self, agent_index):
        odom_data = self.odom_subscribers[agent_index].odom_data

        return Utils.get_position_from_odom_data(odom_data)

    def check_done(self):
        done_n = []
        for i in range(self.agent_count):
            done = self.check_done_id(i)
            done_n.append(done)

        return done_n

    def check_done_id(self, agent_index):
        if Utils.get_distance_to_goal((self.odom_subscribers[agent_index].odom_data.pose.pose.position.x, self.odom_subscribers[agent_index].odom_data.pose.pose.position.y), self.goal_position) < GOAL_REACHED_THRESHOLD:
            self.node.get_logger().info(
                f"Goal reached. Distance to goal: {Utils.get_distance_to_goal((self.odom_subscribers[agent_index].odom_data.pose.pose.position.x, self.odom_subscribers[agent_index].odom_data.pose.pose.position.y), self.goal_position)}")
            return True

        if min(self.scan_subscribers[agent_index].laser_ranges) < OBSTACLE_COLLISION_THRESHOLD:
            self.node.get_logger().info(
                f"Collision detected. minRange: {min(self.scan_subscribers[agent_index].laser_ranges)}")
            return True

        return False

    def get_reward(self, done_n):
        rewards = []
        for i in range(self.agent_count):
            reward_i = self.get_reward_id(done_n[i], i)
            rewards.append(reward_i)

        return rewards

    def get_reward_id(self, done, agent_index: int):
        r_arrive = 200
        r_collision = -200
        k = 5

        distance_to_goal = Utils.get_distance_to_goal(
            (self.odom_subscribers[agent_index].odom_data.pose.pose.position.x, self.odom_subscribers[agent_index].odom_data.pose.pose.position.y), self.goal_position)

        reached_goal = distance_to_goal < GOAL_REACHED_THRESHOLD
        collision = min(
            self.scan_subscribers[agent_index].laser_ranges) < OBSTACLE_COLLISION_THRESHOLD

        if done:
            if reached_goal:
                return r_arrive
            if collision:
                return r_collision

        total_aproach_reward = 0
        for i, _ in enumerate(self.odom_subscribers[agent_index].odom_data):
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
