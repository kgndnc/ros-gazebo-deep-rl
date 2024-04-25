#! /usr/bin/env python3

# @desc Number counter
#
# listen topic = /number
# publish to topic = /number_count


from typing import List
import rclpy
from rclpy.node import Node
from rclpy.node import Parameter

from example_interfaces.msg import Int64
from example_interfaces.srv import SetBool


class NumberCounter(Node):
    def __init__(self, node_name: str, *, context: rclpy.Context = None, cli_args: List[str] = None, namespace: str = None, use_global_arguments: bool = True, enable_rosout: bool = True, start_parameter_services: bool = True, parameter_overrides: List[Parameter] = None, allow_undeclared_parameters: bool = False, automatically_declare_parameters_from_overrides: bool = False) -> None:
        super().__init__(node_name, context=context, cli_args=cli_args, namespace=namespace, use_global_arguments=use_global_arguments, enable_rosout=enable_rosout, start_parameter_services=start_parameter_services,
                         parameter_overrides=parameter_overrides, allow_undeclared_parameters=allow_undeclared_parameters, automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides)

        self.number_subscriber_ = self.create_subscription(
            Int64, "number", self.number_callback, 10)
        self.number_count_publisher_ = self.create_publisher(
            Int64, "number_count", 10)

        self.reset_counter_service_ = self.create_service(SetBool, "reset_counter",
                                                          self.callback_reset_counter)

        self.counter_ = Int64()

    """
    (SrvTypeRequest@create_service, SrvTypeResponse@create_service) 
    -> SrvTypeResponse@create_service
    """

    def callback_reset_counter(self, request, response):
        self.get_logger().info(f"Processing request...")

        shouldReset = request.data
        self.get_logger().info(str(request.data))

        if shouldReset:
            self.counter_.data = 0

        response.message = "Processed your request."
        response.success = True
        return response

    def number_callback(self, msg: Int64):
        # self.get_logger().info(f"I received this message: {msg.data}")
        self.counter_.data += msg.data

        self.number_count_publisher_.publish(self.counter_)


def main(args=None):
    rclpy.init(args=args)
    node = NumberCounter("number_counter")
    rclpy.spin(node)
    rclpy.shutdown()
