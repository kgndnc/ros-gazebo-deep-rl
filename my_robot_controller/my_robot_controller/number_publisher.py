#! /usr/bin/env python3

# @desc Number publisher
#
# topic = /number


import rclpy
from rclpy.node import Node

from example_interfaces.msg import Int64


class NumberPublisher(Node):
    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)
        self.num_publisher = self.create_publisher(Int64, "number", 10)
        self.timer = self.create_timer(1.0, self.num_publisher_callback)

    def num_publisher_callback(self):
        msg = Int64()
        msg.data = 2

        self.num_publisher.publish(msg)

        return


def main(args=None):
    rclpy.init(args=args)
    node = NumberPublisher("number_publisher")
    rclpy.spin(node)
    rclpy.shutdown()
