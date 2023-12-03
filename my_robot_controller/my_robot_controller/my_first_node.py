#! /usr/bin/env python3

import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__("first_node")
        self.get_logger().info("<<ROS 2>>")
        self.counter_ = 0

        self.create_timer(1.0, self.timer_callback)


    def timer_callback(self):
        self.get_logger().info(f"Hello {self.counter_}")
        self.counter_ += 1

        






def main(args=None):
    # initialize ros2 communications and features
    rclpy.init(args=args)

    node = MyNode()
    # this node will be alive until killed 
    rclpy.spin(node)

    # this should be last line
    rclpy.shutdown()

if __name__ == "__main__":
    main()
