import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command

def generate_launch_description():
    pkg = get_package_share_directory('puzzle_sim_pb')
    xacro_file = os.path.join(pkg, 'xacro', 'puzzlebot.urdf.xacro')

    robot_desc = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)

    return LaunchDescription([
        ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
            output='screen'
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_desc, 'use_sim_time': True}]
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            parameters=[{'use_sim_time': True}]
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description', '-entity', 'puzzlebot',
                    '-x', '-1.10', '-y', '0.15', '-z', '0.0'],
            output='screen'
        ),  
        Node(
            package='rviz2',
            executable='rviz2',
            output='screen'
        ),
    ])