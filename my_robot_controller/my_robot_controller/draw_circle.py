#! /usr/bin/env python3

# @desc Creation of publisher
#
# Manipulate the movement of turtlesim_node in turtlesim
# 
# topic = /turtle1/cmd_vel
# 
# To see Twist message structure type `ros2 interface show geometry_msgs/msg/Twist`


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class DrawCircleNode(Node):
    def __init__(self):
        super().__init__("draw_circle")
        self.cmd_vel_pub_ = self.create_publisher(Twist, "turtle1/cmd_vel", 10)
        self.get_logger().info("Draw circle node has been started")
        self.timer = self.create_timer(0.5, self.send_velocity_command)

    
    def send_velocity_command(self):
        msg = Twist()
        msg.linear.x = 2.0
        msg.angular.z = 1.0
        # the publisher we've created
        self.cmd_vel_pub_.publish(msg)
        

def main (args=None):
    rclpy.init(args=args)
    node = DrawCircleNode()
    rclpy.spin(node)
    rclpy.shutdown()