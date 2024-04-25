#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from turtlesim.srv import Spawn
import random
from math import pi
from functools import partial

"""
Call the /spawn service to create a new turtle (choose random coordinates between 0.0 and 11.0 for both x and y),
and call the /kill service to remove a turtle from the screen. 
Both those services are already advertised by the turtlesim_node.
"""


class TurtleSpawner(Node):
    def __init__(self):
        super().__init__("turtle_spawner")
        self.alive_turtles_ = []

        self.create_timer(2.0, self.call_spawn_service)

    def call_spawn_service(self):
        client = self.create_client(Spawn, "spawn")

        # float32 x
        # float32 y
        # float32 theta
        # string name # Optional.  A unique name will be created and returned if this is empty

        new_turtle = {'x': random.uniform(0.0, 11.0), 'y': random.uniform(
            0.0, 11.0), 'theta': random.uniform(0.0, pi)}

        request = Spawn.Request()
        request.x = new_turtle["x"]
        request.y = new_turtle["y"]
        request.theta = new_turtle["theta"]

        future = client.call_async(request=request)

        future.add_done_callback(
            partial(self.callback_spawn_done, new_turtle=new_turtle))

    def callback_spawn_done(self, future, new_turtle):
        try:
            response = future.result()
            # self.get_logger().info(
            #     f"/spawn service has been called, here is the response: {str(response)}")

            new_turtle["name"] = response.name
            self.alive_turtles_.append(new_turtle)

            self.get_logger().info(
                f"List of alive turtles: {str(self.alive_turtles_)}")
        except Exception as e:
            self.get_logger().error("Service call failed %r" % (e,))


def main(args=None):
    rclpy.init(args=args)
    node = TurtleSpawner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
