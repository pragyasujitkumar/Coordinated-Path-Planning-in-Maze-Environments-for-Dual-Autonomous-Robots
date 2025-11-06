#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('maze_solver')
    world = os.path.join(pkg_dir, 'world_easy.world')
    drone = os.path.join(pkg_dir, 'iris.urdf')
    bot = os.path.join(pkg_dir, 'bot.urdf')

    # Drone spawn height
    spawn_height = 10.0

    return LaunchDescription([
        
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            arguments=[bot]),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            arguments=[bot]),
        
        # Launch Gazebo with maze world
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world, '-s', 'libgazebo_ros_factory.so'],
            output='screen'
        ),
        
        # Wait 3s for Gazebo to initialize, then spawn drone
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    arguments=[
                        '-entity', 'iris',
                        '-file', drone,
                        '-x', '0',
                        '-y', '1.5',
                        '-z', str(spawn_height)
                    ],
                    output='screen'
                )
            ]
        ),
        
        # Wait additional 8s for maze_solver to publish path, then spawn bot
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='maze_solver',
                    executable='spawn_bot',  # your path-based spawner
                    name='spawn_bot',
                    output='screen'
                )
            ]
        ),
    ])
