#! /usr/bin/env python3

# @desc Creation of publisher
#
# topic = /turtle1/pose

import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose


class PosePublisherNode(Node):
    def __init__(self):
        super().__init__("pose_publisher")       

        self.pub_ = self.create_publisher(Pose, "/turtle1/pose", 10)
        self.timer = self.create_timer(0.5, self.send_pose_command)

        self.x_counter = 1.0
    

                    
    def send_pose_command(self):
        msg = Pose()
        self.x_counter = self.x_counter + 0.1
        msg.x = self.x_counter
        msg.y = 1.2
        # msg = 0

        self.pub_.publish(msg)
        
        

    """
        self.cmd_vel_pub_ = self.create_publisher(Twist, "turtle1/cmd_vel", 10)
        self.get_logger().info("Draw circle node has been started")
        self.timer = self.create_timer(0.5, self.send_velocity_command)
    """
    
    

def main (args=None):
    rclpy.init(args=args)
    node = PosePublisherNode()
    rclpy.spin(node)
    rclpy.shutdown()