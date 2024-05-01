#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from threading import Timer
from functools import partial

from tutorial_interfaces.srv import CatchTurtle


class TurtleController(Node):
    def __init__(self):
        super().__init__("turtle_controller")

        self.catch_turtle_client_ = self.create_client(
            CatchTurtle, "catch_turtle")

        self.call_catch_turtle_service()

    def call_catch_turtle_service(self):
        self.get_logger().info("Calling catch_turtle service")
        request = CatchTurtle.Request()
        request.turtle_name = "turtle2"

        future = self.catch_turtle_client_.call_async(request=request)
        future.add_done_callback(partial(self.callback_catch_done))

    def callback_catch_done(self, future):

        response = future.result()

        self.get_logger().info(
            f"Callback catch done here's the result {response.success}")


def main(args=None):
    rclpy.init(args=args)
    node = TurtleController()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
