#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from threading import Timer
from functools import partial
import random
import math

from tutorial_interfaces.msg import TurtleArray, Turtle
from tutorial_interfaces.srv import CatchTurtle
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist


class TurtleController(Node):
    def __init__(self):
        super().__init__("turtle_controller")

        self.current_target_: Turtle = None
        self.pose_ = Pose()

        self.catch_turtle_client_ = self.create_client(
            CatchTurtle, "catch_turtle")

        self.alive_turtles_subscribe_ = self.create_subscription(
            TurtleArray, "alive_turtles", self.callback_alive_turtles, 10)

        self.create_subscription(Pose, "turtle1/pose", self.callback_pose, 10)

        self.create_timer(1/60, self.control_loop)

        self.cmd_vel_publisher_ = self.create_publisher(
            Twist, "turtle1/cmd_vel", 10)

        # self.call_catch_turtle_service(name="turtle_name")

    def control_loop(self):
        if self.current_target_ and self.pose_:

            delta_x = self.current_target_.x - self.pose_.x
            delta_y = self.current_target_.y - self.pose_.y

            distance = math.sqrt(delta_x ** 2 + delta_y ** 2)

            msg = Twist()

            tolerance = 0.5

            # P controller

            if distance > tolerance:

                # position
                msg.linear.x = 2 * distance

                # orientation
                goal_theta = math.atan2(delta_y, delta_x)
                delta_theta = goal_theta - self.pose_.theta

                if delta_theta > math.pi:
                    delta_theta -= 2*math.pi
                elif delta_theta < -math.pi:
                    delta_theta += 2*math.pi

                msg.angular.z = 6 * delta_theta

            else:
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                self.call_catch_turtle_service(name=self.current_target_.name)
                self.current_target_ = None

            self.cmd_vel_publisher_.publish(msg)

        # if collision with current_target:
        #   self.call_catch_turtle_service(name= current_target.name)
        #   self.current_target = None
        #

    def callback_pose(self, msg: Pose):
        self.pose_ = msg

    def callback_alive_turtles(self, msg: TurtleArray):
        if not self.current_target_ and msg.turtles:
            # pick one of turtles
            self.current_target_ = self.pick_turtle(msg.turtles)
            self.get_logger().info(
                f"Current target set {self.current_target_}")

    def pick_turtle(self, turtles: list[Turtle]):
        pick_closest = True

        if pick_closest:
            smallest_index = 0
            smallest_dist = calculate_distance(
                self.pose_.x, self.pose_.y, turtles[0].x,  turtles[0].y)

            for i, turtle in enumerate(turtles):
                dist = calculate_distance(
                    self.pose_.x, self.pose_.y, turtle.x, turtle.y)

                if dist < smallest_dist:
                    smallest_dist = dist
                    smallest_index = i

            return turtles[smallest_index]

        else:
            return random.choice(turtles)

    def call_catch_turtle_service(self, name):
        self.get_logger().info("Calling catch_turtle service")
        request = CatchTurtle.Request()

        request.turtle_name = name

        future = self.catch_turtle_client_.call_async(request=request)
        future.add_done_callback(partial(self.callback_catch_done))

    def callback_catch_done(self, future):

        response = future.result()

        self.get_logger().info(
            f"Callback catch done here's the result {response.success}")

    # def get_turtles_distance(self,turtle_1: Turtle, turtle_2: Turtle):
    #     return calculate_distance(turtle_1.x, turtle_1.y, turtle_2.x, turtle_2.y)


def calculate_distance(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    dist = math.sqrt(delta_x**2 + delta_y**2)
    return dist


def main(args=None):
    rclpy.init(args=args)
    node = TurtleController()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
