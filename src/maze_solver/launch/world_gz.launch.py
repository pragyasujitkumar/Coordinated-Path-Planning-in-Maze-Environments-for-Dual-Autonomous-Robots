import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    package_share_dir = get_package_share_directory("maze_solver")
    world_file = os.path.join(package_share_dir, "world_easy.world")
    
    return LaunchDescription([
        # Launch Gazebo with the world file only
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world_file, '-s', 'libgazebo_ros_factory.so'],
            output='screen',
        ),
    ])