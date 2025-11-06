#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
import math


class HoverController(Node):
    def __init__(self):
        super().__init__('hover_controller')
        
        # Publisher for velocity commands
        self.vel_pub = self.create_publisher(Twist, '/iris/cmd_vel', 10)
        
        # Subscribe to Gazebo model states to get actual position
        self.state_sub = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.state_callback,
            10
        )
        
        # Control parameters
        self.target_height = 6.0  # Must match spawn height
        self.current_height = 0.0
        self.drone_found = False
        
        # PID controller parameters for altitude
        self.kp = 2.0  # Proportional gain
        self.ki = 0.1  # Integral gain
        self.kd = 0.5  # Derivative gain
        
        self.error_sum = 0.0
        self.last_error = 0.0
        self.last_time = self.get_clock().now()
        
        # Control loop timer (50Hz)
        self.control_timer = self.create_timer(0.02, self.control_loop)
        
        self.get_logger().info('Hover Controller Started')
        self.get_logger().info(f'Target altitude: {self.target_height}m')
    
    def state_callback(self, msg):
        """Get drone position from Gazebo"""
        try:
            # Find iris model in the states
            if 'iris' in msg.name:
                idx = msg.name.index('iris')
                self.current_height = msg.pose[idx].position.z
                self.drone_found = True
        except (ValueError, IndexError) as e:
            if not self.drone_found:
                self.get_logger().warn('Drone not found in Gazebo states', throttle_duration_sec=5.0)
    
    def control_loop(self):
        """PID control loop to maintain altitude"""
        if not self.drone_found:
            return
        
        # Calculate error
        error = self.target_height - self.current_height
        
        # Time delta
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        
        if dt <= 0:
            return
        
        # PID calculations
        self.error_sum += error * dt
        error_diff = (error - self.last_error) / dt
        
        # Calculate control output
        output = (self.kp * error + 
                 self.ki * self.error_sum + 
                 self.kd * error_diff)
        
        # Limit output
        max_vel = 2.0
        output = max(-max_vel, min(max_vel, output))
        
        # Publish velocity command
        cmd = Twist()
        cmd.linear.z = output
        self.vel_pub.publish(cmd)
        
        # Store last error
        self.last_error = error
        
        # Log status
        if abs(error) > 0.1:
            self.get_logger().info(
                f'Altitude: {self.current_height:.2f}m | Error: {error:.2f}m | Output: {output:.2f}',
                throttle_duration_sec=1.0
            )


def main(args=None):
    rclpy.init(args=args)
    controller = HoverController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down hover controller...')
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()