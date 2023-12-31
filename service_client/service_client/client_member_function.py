import sys

from example_interfaces.srv import AddTwoInts
from tutorial_interfaces.srv import AddThreeInts
import rclpy
from rclpy.node import Node


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__("minimal_client_async")
        self.cli_ = self.create_client(AddThreeInts, 'add_three_ints')

        while not self.cli_.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.req_ = AddThreeInts.Request()

    def send_request(self, a, b, c):
        self.req_.a = a
        self.req_.b = b
        self.req_.c = c
        self.future = self.cli_.call_async(self.req_)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main():
    rclpy.init()

    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(
        int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

    minimal_client.get_logger().info(
        'Result of service: for %d + %d + %d = %d' %
        (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), response.sum))

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
