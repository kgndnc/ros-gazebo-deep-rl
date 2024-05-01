#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from turtlesim.srv import Spawn, Kill
import random
from math import pi
from functools import partial
from tutorial_interfaces.msg import Turtle, TurtleArray
from tutorial_interfaces.srv import CatchTurtle


"""
Call the /spawn service to create a new turtle (choose random coordinates between 0.0 and 11.0 for both x and y),
and call the /kill service to remove a turtle from the screen. 
Both those services are already advertised by the turtlesim_node.
"""


class TurtleSpawner(Node):
    def __init__(self):
        super().__init__("turtle_spawner")

        # default value 1.0
        self.declare_parameter('spawn_frequency', 1.0)
        self.spawn_frequency_ = self.get_parameter('spawn_frequency').value

        self.create_timer(1.0/self.spawn_frequency_, self.call_spawn_service)

        self.alive_turtles_ = []

        self.alive_turtles_publisher_ = self.create_publisher(
            TurtleArray, "alive_turtles", 10)

        self.catch_turtle_service_ = self.create_service(
            CatchTurtle, "catch_turtle", self.catch_turtle)

    """
    Handle a service server to “catch” a turtle, which means to call the 
    /kill service and remove the turtle from the array of alive turtles.
    """

    def catch_turtle(self, request, response):
        self.get_logger().info("Inside catch_turtle service")
        print(request)

        client = self.create_client(Kill, "kill")
        req = Kill.Request()
        req.name = request.turtle_name
        future = client.call_async(request=req)
        future.add_done_callback(partial(
            self.callback_kill_done, killed_turtle_name=req.name
        ))

        response.success = True
        return response

    def callback_kill_done(self, future, killed_turtle_name):
        self.get_logger().info(f"Killing turtle {killed_turtle_name}")

        try:
            response = future.result()

            index = None

            for i, turtle in enumerate(self.alive_turtles_):
                if turtle.name == killed_turtle_name:
                    index = i
                    break

            if index != None:
                del self.alive_turtles_[index]

            self.publish_alive_turtles()

        except Exception as e:
            self.get_logger().error("Service call failed %r" % (e,))

        return

    """
    Publish the list of currently alive turtles with coordinates on a topic /alive_turtles.
    """

    def publish_alive_turtles(self):
        msg = TurtleArray()
        msg.turtles = self.alive_turtles_
        self.alive_turtles_publisher_.publish(msg)

    def call_spawn_service(self):
        client = self.create_client(Spawn, "spawn")

        # float32 x
        # float32 y
        # float32 theta
        # string name # Optional.  A unique name will be created and returned if this is empty

        new_turtle = {'x': random.uniform(0.5, 10.5), 'y': random.uniform(
            0.5, 10.5), 'theta': random.uniform(0.0, pi)}

        request = Spawn.Request()
        request.x = new_turtle["x"]
        request.y = new_turtle["y"]
        request.theta = new_turtle["theta"]

        future = client.call_async(request=request)

        future.add_done_callback(
            partial(self.callback_spawn_done, new_turtle=new_turtle)
        )

    def callback_spawn_done(self, future, new_turtle: Turtle):
        try:
            response = future.result()

            added_turtle = Turtle()
            added_turtle.name = response.name
            added_turtle.x = new_turtle["x"]
            added_turtle.y = new_turtle["y"]
            added_turtle.theta = new_turtle["theta"]

            self.alive_turtles_.append(added_turtle)
            self.publish_alive_turtles()

        except Exception as e:
            self.get_logger().error("Service call failed %r" % (e,))


def main(args=None):
    rclpy.init(args=args)
    node = TurtleSpawner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
