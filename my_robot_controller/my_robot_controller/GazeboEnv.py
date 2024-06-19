import time
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from gymnasium import spaces
from gymnasium import Env
from rclpy.node import Node
import numpy as np
import math
from typing import Any, SupportsFloat
from Utils import Utils


GOAL_REACHED_THRESHOLD = 1.0
OBSTACLE_COLLISION_THRESHOLD = 0.7
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
