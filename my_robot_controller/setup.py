# TODO: Implement service-client


from setuptools import find_packages, setup

package_name = 'my_robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kgndnc',
    maintainer_email='hkagandnc@gmail.com',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "test_node = my_robot_controller.my_first_node:main",
            "draw_circle = my_robot_controller.draw_circle:main",
            "pose_subscribe = my_robot_controller.pose_subscriber:main",
            "pose_publish = my_robot_controller.pose_publisher:main",
            "number_publish = my_robot_controller.number_publisher:main",
            "number_count = my_robot_controller.number_counter:main",
            "reset_count = my_robot_controller.reset_counter_client:main",
            "q_learning = my_robot_controller.q_learning:main",
            "ddpg = my_robot_controller.ddpg:main",
            "DDPG = my_robot_controller.DDPG:main",
            "common_def = my_robot_controller.common_definitions:main",
            "updated_keras_ddpg = my_robot_controller.updated_keras_ddpg:main",
        ],
    },
)
