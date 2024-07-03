import math
import numpy as np
from numpy.typing import NDArray


class Utils:
    @staticmethod
    def discretize(value, min_value, max_value, num_bins):
        return int((value - min_value) / (max_value - min_value) * num_bins)

    @staticmethod
    def get_angle_between_points(ref_point, point_1_heading, target_point):

        target_vector = [target_point[0] - ref_point[0],
                         target_point[1] - ref_point[1]]

        target_angle = math.atan2(target_vector[1], target_vector[0])

        angle = target_angle - point_1_heading
        return angle

    @staticmethod
    def get_distance_between_points(point_1: tuple[float, float], point_2: tuple[float, float]):
        x_1, y_1 = point_1
        x_2, y_2 = point_2

        dist = math.sqrt(((y_2 - y_1)**2) + ((x_2 - x_1)**2))
        return dist

    @staticmethod
    def get_distance_to_goal(robot_position, goal_position):
        return Utils.get_distance_between_points(robot_position, goal_position)

    @staticmethod
    def get_angle_to_goal(robot_position, robot_orientation, goal_position):

        # print("Calculating angle")

        goal_vector = [goal_position[0] - robot_position[0],
                       goal_position[1] - robot_position[1]]
        goal_angle = math.atan2(goal_vector[1], goal_vector[0])

        # Assuming robot_orientation is given as yaw angle (heading)
        angle_to_goal = goal_angle - robot_orientation

        # Normalize the angle to be within [-pi, pi]
        angle_to_goal = (angle_to_goal + math.pi) % (2 * math.pi) - math.pi

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
    def reduce_lidar_samples(lidar_data: NDArray, num_sectors=20):

        samples_per_sector = len(lidar_data) // num_sectors

        # Reshape the data into sectors
        sectors = lidar_data[:samples_per_sector *
                             num_sectors].reshape(num_sectors, samples_per_sector)

        reduced_data = np.min(sectors, axis=1)

        return reduced_data

    @staticmethod
    def discretize_position(position, bounds, grid_size):
        """
        Discretizes a continuous position into a grid index.

        Args:
        - position: The continuous position value (x or y).
        - bounds: A tuple (min_value, max_value) representing the bounds of the environment.
        - grid_size: The number of discretes in the grid.

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
    def get_position_from_odom_data(odom):
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        return (x, y)

    @staticmethod
    def get_min_distances_from_slices(laser_data, num_slices=4):
        """
        Divide the laser data into slices and take the minimum distance from each slice.

        Args:
        - laser_data: Array of laser scan distances.
        - num_slices: Number of slices to divide the laser data into (default is 4).

        Returns:
        - List of minimum distances from each slice.
        """
        slice_size = len(laser_data) // num_slices
        min_distances = []

        for i in range(num_slices):
            start_index = i * slice_size
            end_index = start_index + slice_size
            slice_min = min(laser_data[start_index:end_index])
            # slice_min = round(slice_min, 2)
            slice_min = round(slice_min, 0)
            min_distances.append(slice_min)

        return min_distances
