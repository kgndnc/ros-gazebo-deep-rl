import rclpy
from gazebo_msgs.srv import SpawnEntity


def spawn_entity(node):
    client = node.create_client(SpawnEntity, 'spawn_entity')
    request = SpawnEntity.Request()

    # Set the name of the model and its pose
    request.name = 'my_robot_model'
    request.xml = open('path/to/your/my_robot_model.sdf', 'r').read()
    request.robot_namespace = ''
    request.initial_pose.position.x = 0.0
    request.initial_pose.position.y = 0.0
    request.initial_pose.position.z = 0.0
    request.initial_pose.orientation.x = 0.0
    request.initial_pose.orientation.y = 0.0
    request.initial_pose.orientation.z = 0.0
    request.initial_pose.orientation.w = 1.0

    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Service not available, waiting again...')

    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('spawn_model_node')

    try:
        spawn_entity(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
