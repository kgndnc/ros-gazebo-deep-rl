```python
def execute_callback(self, goal_handle):
    """Execute the goal."""
    self.get_logger().info('Executing goal...')

    rotation_request = goal_handle.request.rotation

    with self._odom_lock:
        curr_orientation = Utils.euler_from_quaternion(
            self.robot_controller.odom_data.pose.pose.orientation)

    target_orientation = curr_orientation + rotation_request

    # Normalize the target orientation to be within -pi to pi
    target_orientation = (target_orientation + 3.14159) % (2 * 3.14159) - 3.14159

    remaining_orientation = target_orientation - curr_orientation

    # feedback
    feedback_msg = Rotate.Feedback()
    feedback_msg.remaining_rotation = remaining_orientation

    Kp = 1.0  # Proportional gain

    # Start executing the action
    while abs(remaining_orientation) > 0.01:  # Use a smaller threshold for better precision
        # If goal is flagged as no longer active (ie. another goal was accepted),
        # then stop executing
        if not goal_handle.is_active:
            self.get_logger().info('Goal aborted')
            result = Rotate.Result()
            result.success = False
            return result

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            self.get_logger().info('Goal canceled')
            result = Rotate.Result()
            result.success = False
            return result

        # Calculate the remaining orientation again
        with self._odom_lock:
            curr_orientation = Utils.euler_from_quaternion(
                self.robot_controller.odom_data.pose.pose.orientation)
            remaining_orientation = target_orientation - curr_orientation

        # Normalize the remaining orientation
        remaining_orientation = (remaining_orientation + 3.14159) % (2 * 3.14159) - 3.14159

        # Calculate the angular velocity using a proportional controller
        cmd_vel = Twist()
        cmd_vel.angular.z = Kp * remaining_orientation
        cmd_vel.angular.z = max(min(cmd_vel.angular.z, 0.8), -0.8)  # Cap the angular velocity

        self.cmd_vel_pub_.publish(cmd_vel)

        # Update feedback
        feedback_msg.remaining_rotation = remaining_orientation

        self.get_logger().info(
            'Publishing feedback: {0}'.format(feedback_msg))

        # Publish the feedback
        goal_handle.publish_feedback(feedback_msg)

        # Sleep for demonstration purposes
        time.sleep(0.02)  # Increase the frequency

    with self._goal_lock:
        if not goal_handle.is_active:
            self.get_logger().info('Goal aborted')
            result = Rotate.Result()
            result.success = False
            return result

        # goal_handle.succeed()

    #  stop the movement
    stop_cmd = Twist()
    stop_cmd.linear.x = 0.0
    stop_cmd.angular.z = 0.0
    self.cmd_vel_pub_.publish(stop_cmd)

    # Populate result message
    goal_handle.succeed()
    result = Rotate.Result()
    result.success = True

    self.get_logger().info('Returning result: {0}'.format(result.success))

    return result
```
