#! /usr/bin/env python3


from typing import List
import rclpy
from rclpy.node import Node
from rclpy.node import Parameter

from example_interfaces.msg import Int64
from example_interfaces.srv import SetBool


class ResetCounterClient(Node):
    def __init__(self, node_name: str, *, context: rclpy.Context = None, cli_args: List[str] = None, namespace: str = None, use_global_arguments: bool = True, enable_rosout: bool = True, start_parameter_services: bool = True, parameter_overrides: List[Parameter] = None, allow_undeclared_parameters: bool = False, automatically_declare_parameters_from_overrides: bool = False) -> None:
        super().__init__(node_name, context=context, cli_args=cli_args, namespace=namespace, use_global_arguments=use_global_arguments, enable_rosout=enable_rosout, start_parameter_services=start_parameter_services,
                         parameter_overrides=parameter_overrides, allow_undeclared_parameters=allow_undeclared_parameters, automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides)

        self.call_reset_counter_service()

    def call_reset_counter_service(self):
        client = self.create_client(SetBool, "reset_counter")

        request = SetBool.Request()
        request.data = True

        future = client.call_async(request=request)
        future.add_done_callback(self.reset_counter_done)

    def reset_counter_done(self, future):
        try:
            response = future.result()
            self.get_logger().info(
                f"Service has been called, here is the response: {str(response)}")
        except Exception as e:
            self.get_logger().error("Service call failed %r" % (e,))


def main(args=None):
    rclpy.init(args=args)
    node = ResetCounterClient("reset_counter_client")
    rclpy.spin(node)
    rclpy.shutdown()
