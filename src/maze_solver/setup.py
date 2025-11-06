from setuptools import setup
from glob import glob
import os

package_name = 'maze_solver'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*')),
        (os.path.join('share', package_name), glob('urdf/*')),
        (os.path.join('share', package_name), glob('meshes/drone/*')),
        (os.path.join('share', package_name), glob('world/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='asha',
    maintainer_email='asha@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drone_teleop = maze_solver.drone_keyboard_teleop:main',
            'hover_controller = maze_solver.hover_controller:main',
            'maze_solver_node = maze_solver.maze_solver_node:main',
            'camera_viewer = maze_solver.camera_viewer:main',
            'move = maze_solver.adaptive_maze_solver:main',
            'maze_follower = maze_solver.maze_path_follower:main',
            'try = maze_solver.try:main',
            'spawn_bot = maze_solver.spawn_bot:main',
        ],
    },
)