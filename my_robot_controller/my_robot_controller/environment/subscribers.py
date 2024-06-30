from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
from config import AGENT_COUNT, LIDAR_SAMPLE_SIZE


class OdomSubscriber(Node):
    def __init__(self, namespace: str, robot_index: int):
        super().__init__('odom_subscriber_' + namespace)
        self.subscription = self.create_subscription(
            Odometry,
            "/" + namespace + '/odom',
            self.odom_callback,
            10)
        self.robot_index = robot_index
        self.odom_data = Odometry()

    def odom_callback(self, msg: Odometry):
        self.odom_data = msg


class ScanSubscriber(Node):
    def __init__(self, namespace: str, robot_index: int):
        super().__init__('scan_subscriber_' + namespace)
        self.subscription = self.create_subscription(
            LaserScan, "/" + namespace + "/scan", self.scan_callback, 10)
        self.robot_index = robot_index
        self.laser_ranges = np.zeros(LIDAR_SAMPLE_SIZE)

    def scan_callback(self, msg: LaserScan):
        self.laser_ranges = np.array(msg.ranges)
