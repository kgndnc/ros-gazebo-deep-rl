from numpy.typing import NDArray
import random
from collections import namedtuple
from gymnasium import Env, spaces
import numpy as np
import time
from typing import List, Tuple
from rclpy.node import Node
from gymnasium import Env, spaces
from Utils import Utils
import math
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState, GetEntityState, SetModelState, SetModelConfiguration, SetLinkState, SetLinkProperties, SetPhysicsProperties
from subscribers import OdomSubscriber, ScanSubscriber
from config import GOAL_REACHED_THRESHOLD, OBSTACLE_COLLISION_THRESHOLD, LIDAR_SAMPLE_SIZE, SAVE_INTERVAL, AGENT_COUNT, MAX_LIDAR_RANGE, RTF


# (done, collision, target, min_laser_reading)
DoneTuple = namedtuple(
    'DoneTuple', ['done', 'collision', 'target', 'min_laser_reading'])


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


class GazeboEnvMultiAgent(Env):

    def __init__(self, odom_subscribers: List[OdomSubscriber], scan_subscribers: List[ScanSubscriber], goal_position=(0., 0.), test=False):
        super(GazeboEnvMultiAgent, self).__init__()

        self.odom_subscribers = odom_subscribers
        self.scan_subscribers = scan_subscribers
        self.test = test

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

        """
        /delete_entity: type=gazebo_msgs/srv/DeleteEntity (string name: Name of the Gazebo entity to be deleted.)
        /spawn_entity: type=gazebo_msgs/srv/SpawnEntity (name, xml, initial_pose)
        """

        self.unpause = self.node.create_client(Empty, "/unpause_physics")
        self.pause = self.node.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.node.create_client(Empty, "/reset_world")
        self.spawn_goal_client = self.node.create_client(
            SpawnEntity, "/spawn_entity")
        self.delete_goal_client = self.node.create_client(
            DeleteEntity, "/delete_entity")

        # /gazebo/set_entity_state
        self.set_entity_state = self.node.create_client(
            SetEntityState, "/gazebo/set_entity_state")

        self.req = Empty.Request
        self.goal_position = goal_position

        self.last_actions = {agent_index: (0.0, 0.0)
                             for agent_index in range(agent_count)}
        self.prev_distances_to_goal = [Utils.get_distance_to_goal(self.get_robot_position(
            agent_index), self.goal_position) for agent_index in range(agent_count)]

    goal_positions = [(-6.5, -6.0), (0.43, 1.58), (-8.061270, 1.007540),
                      (-4.224874, 6.551670), (7.500443, 8.963343),
                      (-1.097499, 11.650926), (-8.499418, -9.187337),
                      (-1.944295, -8.582796), (-4.369452, 1.338763)]

    def raise_robot_model(self, robot_index: int):

        msg = SetLinkProperties.Request()
        msg.link_name = "my_robot_1::base_footprint"
        msg.gravity_mode = False

        model_name: str = f"my_robot_{robot_index+1}"
        print(f"Raising {model_name}")
        request = SetEntityState.Request()
        request.state.name = model_name

        request.state.pose.position.z = -2.0

        while not self.set_entity_state.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(
                "Waiting for set entity state service to be available")
        try:
            self.set_entity_state.call_async(request)
        except:
            self.node.get_logger().error("/gazebo/set_entity_state service call failed!")

    def change_goal_position(self):

        # TODO: make this dynamic

        goal_position = random.choice(GazeboEnvMultiAgent.goal_positions)

        request = SetEntityState.Request()
        request.state.name = "bookshelf"
        request.state.pose.position.x = goal_position[0]
        request.state.pose.position.y = goal_position[1]

        request.state.pose.orientation.x = 0.0
        request.state.pose.orientation.y = 0.0
        request.state.pose.orientation.z = 0.7089829505001405
        request.state.pose.orientation.w = 0.7052256205641677

        while not self.set_entity_state.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(
                "Waiting for set entity state service to be available")
        try:
            self.set_entity_state.call_async(request)
            self.goal_position = goal_position
            print(f"Goal position changed. New position: {self.goal_position}")
        except:
            self.node.get_logger().error("/gazebo/set_entity_state service call failed!")

    def step(self, action_n: List):

        cmds = []

        for i in range(self.agent_count):
            linear_x = float(action_n[i][0])
            angular_z = float(action_n[i][1])
            msg = Twist()
            msg.linear.x = linear_x
            msg.angular.z = angular_z
            cmds.append(msg)
            self.last_actions[i] = (linear_x, angular_z)

        # print(f"Publishing velocities {action_n}")

        for i in range(self.agent_count):
            cmd = cmds[i]
            self.vel_pubs[i].publish(cmd)

        self.unpause_physics()

        time.sleep(0.2 / RTF)

        self.pause_physics()

        # observation = (lidar ranges, relative params of target, last action)
        observation = self.get_obs()

        # (done_n, collision_n, target_n, min_laser_readings)
        terminated, collision, target, min_laser_readings = self.check_done()

        reward = self.get_reward(
            terminated, collision, target, action_n, min_laser_readings)

        info = {"terminated": terminated,
                "collision": collision, "target": target}

        return observation, reward, terminated, info

    def unpause_physics(self):
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("Unpause physics service not available, waiting")

        try:
            self.unpause.call_async(Empty.Request())
        except:
            self.node.get_logger().error("Unpause physics service call failed!")

    def pause_physics(self):
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("Pause physics service not available, waiting")

        try:
            self.pause.call_async(Empty.Request())
        except:
            self.node.get_logger().error("Pause physics service call failed!")

    def reset(self):
        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("/reset_world service not available, waiting")

        try:
            self.reset_proxy.call_async(Empty.Request())
        except:
            self.node.get_logger().error("/reset_world service call failed!")

        # set bookshelf position
        # self.set_goal_position()

        time.sleep(0.1 / RTF)

        self.change_goal_position()

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

            # reduce 180 samples to 20
            reduced_lidar_ranges = Utils.reduce_lidar_samples(
                self.scan_subscribers[i].laser_ranges, 20)

            normalized_lidar_ranges = reduced_lidar_ranges / max_lidar_range

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

    def get_robot_position(self, agent_index):
        odom_data = self.odom_subscribers[agent_index].odom_data
        return Utils.get_position_from_odom_data(odom_data)

    def check_done(self):
        done_n = []
        collision_n = []
        target_n = []
        min_laser_reading_n = []

        for i in range(self.agent_count):
            done_tuple = self.check_done_id(i)
            done_n.append(done_tuple.done)
            collision_n.append(done_tuple.collision)
            target_n.append(done_tuple.target)
            min_laser_reading_n.append(done_tuple.min_laser_reading)

        return done_n, collision_n, target_n, min_laser_reading_n

    def check_done_id(self, agent_index) -> DoneTuple:
        distance_to_goal = Utils.get_distance_to_goal(
            self.get_robot_position(agent_index), self.goal_position)

        min_laser_reading = min(
            self.scan_subscribers[agent_index].laser_ranges)

        if distance_to_goal < GOAL_REACHED_THRESHOLD:
            self.node.get_logger().info(
                f"Goal reached. Distance to goal: {Utils.get_distance_to_goal((self.odom_subscribers[agent_index].odom_data.pose.pose.position.x, self.odom_subscribers[agent_index].odom_data.pose.pose.position.y), self.goal_position)}")
            return DoneTuple(done=True, collision=False, target=True, min_laser_reading=min_laser_reading)

        if min_laser_reading < OBSTACLE_COLLISION_THRESHOLD:
            self.node.get_logger().info(
                f"Collision detected. minRange: {min_laser_reading}")
            return DoneTuple(done=True, collision=True, target=False, min_laser_reading=min_laser_reading)

        return DoneTuple(done=False, collision=False, target=False,
                         min_laser_reading=min_laser_reading)

    def get_reward(self, done_n, collision_n, target_n, action_n, min_laser_reading_n):
        rewards = []
        # total_approach_reward = self.calculate_approach_reward()
        for i in range(self.agent_count):
            reward_i = self.get_reward_id(
                done_n[i], collision_n[i], target_n[i], action_n[i], min_laser_reading_n[i], i)
            # if reward_i == "TOTAL_APPROACH_REWARD":
            #     reward_i = total_approach_reward
            rewards.append(reward_i)

        # update previous distances
        return rewards

    def get_reward_id(self, done: bool, collision: bool, target: bool, action: Tuple[float, float], min_laser_reading: float, agent_index: int):
        r_arrive = 100.0
        r_collision = -100.0
        k = 50

        distance_to_goal = Utils.get_distance_to_goal(
            (self.odom_subscribers[agent_index].odom_data.pose.pose.position.x, self.odom_subscribers[agent_index].odom_data.pose.pose.position.y), self.goal_position)
        self.prev_distances_to_goal[agent_index] = distance_to_goal

        if done:
            if target:
                return r_arrive
            if collision:
                return r_collision
            else:
                return 0.0

        else:
            # r = v - |ω|
            def r3(x): return 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser_reading) / 2

    def calculate_approach_reward(self):
        k = 50
        total_approach_reward = 0.0
        for i in range(agent_count):
            current_distance_to_goal = Utils.get_distance_to_goal(
                self.get_robot_position(i), self.goal_position)

            approach_dist = self.prev_distances_to_goal[i] - \
                current_distance_to_goal
            approach_dist *= k

            total_approach_reward += approach_dist

        return total_approach_reward

    def close(self):
        self.node.destroy_node()
