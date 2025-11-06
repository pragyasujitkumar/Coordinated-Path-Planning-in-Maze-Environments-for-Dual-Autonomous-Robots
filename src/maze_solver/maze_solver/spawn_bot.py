#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose, Point, Quaternion
import json
import os
import subprocess

class SpawnBot(Node):
    def __init__(self):
        super().__init__('spawn_bot')

        self.bot_xacro = os.path.join(
            os.getenv('HOME'), 'try_rppm_ws/src/maze_solver/urdf/bot.urdf.xacro'
        )
        self.spawned = False

        self.client = self.create_client(SpawnEntity, '/spawn_entity')
        self.path_sub = self.create_subscription(String, '/maze/path_data', self.path_callback, 10)
        self.get_logger().info("Waiting for path data...")

    def path_callback(self, msg):
        if self.spawned:
            return

        self.get_logger().info(f"Received path message: {msg.data}")  # <-- add this

        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("Invalid JSON in path message")
            return

        path = data.get('path', [])
        if not path:
            self.get_logger().warn("Empty path received")
            return

        # Cast to float
        x, y = map(float, path[0])
        z = 0.1

        # Print the coordinates for debugging
        self.get_logger().info(f"Spawning bot at coordinates: x={x}, y={y}, z={z}")

        if not self.client.service_is_ready():
            self.get_logger().info("Spawn service not ready, retrying in 0.5s")
            self.create_timer(0.5, lambda: self.path_callback(msg))
            return

        # Convert XACRO to URDF
        try:
            urdf_xml = subprocess.check_output(['xacro', self.bot_xacro]).decode('utf-8')
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"Failed to convert XACRO: {e}")
            return

        req = SpawnEntity.Request()
        req.name = 'maze_bot'
        req.xml = urdf_xml
        req.robot_namespace = ''
        req.initial_pose = Pose(
            position=Point(x=x, y=y, z=z),
            orientation=Quaternion(x=90.0, y=0.0, z=0.0, w=1.0)
        )

        future = self.client.call_async(req)
        future.add_done_callback(lambda f: self.get_logger().info("Bot spawned successfully!"))
        self.spawned = True


def main(args=None):
    rclpy.init(args=args)
    node = SpawnBot()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
