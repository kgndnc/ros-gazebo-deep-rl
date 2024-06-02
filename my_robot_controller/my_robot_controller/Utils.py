import math


class Utils:
    @staticmethod
    def discretize(value, min_value, max_value, num_bins):
        return int((value - min_value) / (max_value - min_value) * num_bins)

    @staticmethod
    def get_distance_to_goal(robot_position, goal_position):
        return math.sqrt((goal_position[0] - robot_position[0]) ** 2 + (goal_position[1] - robot_position[1]) ** 2)

    @staticmethod
    def get_angle_to_goal(robot_position, robot_orientation, goal_position):
        goal_vector = [goal_position[0] - robot_position[0],
                       goal_position[1] - robot_position[1]]
        goal_angle = math.atan2(goal_vector[1], goal_vector[0])

        # Assuming robot_orientation is given as yaw angle (heading)
        angle_to_goal = goal_angle - robot_orientation
        return angle_to_goal

    @staticmethod
    def euler_from_quaternion(quaternion):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        """
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return yaw_z  # in radians

    @staticmethod
    def discretize_position(position, bounds, grid_size):
        """
        Discretizes a continuous position into a grid index.

        Args:
        - position: The continuous position value (x or y).
        - bounds: A tuple (min_value, max_value) representing the bounds of the environment.
        - grid_size: The number of discrete steps in the grid.

        Returns:
        - The discrete index corresponding to the position.
        """
        min_value, max_value = bounds
        scale = grid_size / (max_value - min_value)
        index = int((position - min_value) * scale)
        # Ensure the index is within bounds
        index = max(0, min(grid_size - 1, index))

        return index

    @staticmethod
    def discretize_odom_data(odom, bounds, x_grid_size, y_grid_size):
        """
        Discretizes the odometry data into a discrete state.

        Args:
        - odom: The odometry message containing the position.
        - bounds: A tuple ((min_x, max_x), (min_y, max_y)) representing the bounds of the environment.
        - grid_size: The number of discrete steps in the grid.

        Returns:
        - A tuple (discrete_x, discrete_y) representing the discrete state.
        """
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y

        discrete_x = Utils.discretize_position(
            x, bounds[0], x_grid_size)
        discrete_y = Utils.discretize_position(
            y, bounds[1], y_grid_size)
        return (discrete_x, discrete_y)

    @staticmethod
    def calculate_remaining_orientation():
        return
        curr_orientation = Utils.euler_from_quaternion(
            odom_data.pose.pose.orientation)
        remaining_orientation = self.target_orientation - curr_orientation
        # Normalize the remaining orientation
        remaining_orientation = (
            remaining_orientation + math.pi) % (2 * math.pi) - math.pi

        return remaining_orientation
