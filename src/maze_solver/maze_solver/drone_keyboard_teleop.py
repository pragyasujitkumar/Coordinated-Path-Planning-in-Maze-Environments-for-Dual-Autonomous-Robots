#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        self.vel_pub = self.create_publisher(Twist, '/iris/cmd_vel', 10)
        
        self.linear_speed = 0.3
        self.angular_speed = 0.5
        
        self.settings = termios.tcgetattr(sys.stdin)
        
        self.get_logger().info('Keyboard Teleop Node Started')
        self.print_instructions()
    
    def print_instructions(self):
        msg = """
        Drone Keyboard Control
        ----------------------
        Moving around:
           w
        a  s  d
        
        w/s : forward/backward
        a/d : left/right
        q/e : rotate left/right
        
        space : stop
        +/- : increase/decrease speed
        
        CTRL-C to quit
        """
        print(msg)
    
    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key
    
    def run(self):
        try:
            while True:
                key = self.get_key()
                cmd = Twist()
                
                if key == 'w':
                    cmd.linear.x = self.linear_speed
                    self.get_logger().info('Moving forward')
                elif key == 's':
                    cmd.linear.x = -self.linear_speed
                    self.get_logger().info('Moving backward')
                elif key == 'a':
                    cmd.linear.y = self.linear_speed
                    self.get_logger().info('Moving left')
                elif key == 'd':
                    cmd.linear.y = -self.linear_speed
                    self.get_logger().info('Moving right')
                elif key == 'q':
                    cmd.angular.z = self.angular_speed
                    self.get_logger().info('Rotating left')
                elif key == 'e':
                    cmd.angular.z = -self.angular_speed
                    self.get_logger().info('Rotating right')
                elif key == ' ':
                    cmd.linear.x = 0.0
                    cmd.linear.y = 0.0
                    cmd.angular.z = 0.0
                    self.get_logger().info('Stopped')
                elif key == '+':
                    self.linear_speed = min(1.0, self.linear_speed + 0.1)
                    self.get_logger().info(f'Speed increased to {self.linear_speed:.1f}')
                    continue
                elif key == '-':
                    self.linear_speed = max(0.1, self.linear_speed - 0.1)
                    self.get_logger().info(f'Speed decreased to {self.linear_speed:.1f}')
                    continue
                elif key == '\x03':  # CTRL-C
                    break
                else:
                    continue
                
                self.vel_pub.publish(cmd)
                
        except Exception as e:
            self.get_logger().error(str(e))
        finally:
            # Stop the drone
            cmd = Twist()
            self.vel_pub.publish(cmd)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


def main(args=None):
    rclpy.init(args=args)
    teleop = KeyboardTeleop()
    
    try:
        teleop.run()
    except KeyboardInterrupt:
        pass
    finally:
        teleop.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()