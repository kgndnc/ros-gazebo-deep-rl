from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    # 3 Nodes

    turtlesim = Node(
        package="turtlesim",
        executable="turtlesim_node"
    )

    turtle_spawner_node = Node(
        package="turtlesim_catch_them_all",
        # console script name, not python file name
        executable="turtle_spawn",
        parameters=[
            {'spawn_frequency': 1.0}
        ]
    )

    turtle_controller_node = Node(
        package="turtlesim_catch_them_all",
        executable="turtle_control",
        parameters=[
            {'catch_closest_turtle_first': True}
        ]
    )

    ld.add_action(turtlesim)
    ld.add_action(turtle_spawner_node)
    ld.add_action(turtle_controller_node)

    return ld
