#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import json
import random

import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion

class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('maze_path_follower')
        
        # Publishers and Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.waypoint_sub = self.create_subscription(String, '/maze/path_data', self.waypoint_callback, 10)
        
        # LiDAR subscriptions
        self.lidar_sub = self.create_subscription(LaserScan, '/gazebo_lidar/out', self.lidar_callback, 10)
        self.lidar_sub2 = self.create_subscription(LaserScan, '/laser_scan', self.lidar_callback, 10)
        self.lidar_sub3 = self.create_subscription(LaserScan, '/lidar', self.lidar_callback, 10)
        
        # Robot state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # LiDAR data
        self.lidar_ranges = []
        self.lidar_angle_min = 0.0
        self.lidar_angle_max = 0.0
        self.lidar_angle_increment = 0.0
        self.lidar_received = False
        
        # Waypoint queue
        self.waypoints = []
        self.current_waypoint = None
        self.navigating = False
        
        # Control parameters
        self.linear_speed = 1.2
        self.angular_speed = 1.0
        self.position_tolerance = 0.25
        self.angle_tolerance = 0.15
        
        # Improved obstacle avoidance parameters
        self.safe_distance = 0.6          # Start slowing down
        self.danger_distance = 0.4        # Active avoidance
        self.critical_distance = 0.3      # Emergency stop
        self.front_angle_range = math.pi / 3  # 60° cone in front
        
        # Obstacle avoidance state
        self.avoiding_obstacle = False
        self.avoidance_direction = 0.0
        self.stuck_counter = 0
        self.max_stuck_count = 20
        self.recovery_mode = False
        
        # Random waypoint generation (disabled by default)
        self.use_random_waypoints = False
        
        # Waypoint filtering - ignore waypoints closer than 1 unit
        self.waypoint_distance_threshold = 2.0
        
        # Timer for navigation loop (20 Hz for more responsive control)
        self.timer = self.create_timer(0.05, self.navigation_loop)
        
        self.get_logger().info("="*60)
        self.get_logger().info("Enhanced Waypoint Navigator with Smart Obstacle Avoidance")
        self.get_logger().info("Filters waypoints closer than 1.0 unit (Euclidean distance)")
        self.get_logger().info("="*60)
    
    def generate_random_waypoints(self):
        """Generate random waypoints including start and end positions"""
        waypoints = []
        
        # Start position (current position or origin)
        start_x, start_y = 0.0, 0.0
        waypoints.append((start_x, start_y))
        
        # Generate random intermediate waypoints
        for _ in range(self.num_random_waypoints - 2):
            x = random.uniform(self.map_bounds['x_min'], self.map_bounds['x_max'])
            y = random.uniform(self.map_bounds['y_min'], self.map_bounds['y_max'])
            waypoints.append((x, y))
        
        # End position (random or specified)
        end_x = random.uniform(self.map_bounds['x_min'], self.map_bounds['x_max'])
        end_y = random.uniform(self.map_bounds['y_min'], self.map_bounds['y_max'])
        waypoints.append((end_x, end_y))
        
        # Filter close waypoints
        self.waypoints = self.filter_waypoints(waypoints)
        
        self.get_logger().info(f"✓ Generated {len(self.waypoints)} random waypoints")
        self.get_logger().info(f"  Start: ({start_x:.2f}, {start_y:.2f})")
        self.get_logger().info(f"  End: ({end_x:.2f}, {end_y:.2f})")
    
    def odom_callback(self, msg):
        """Update robot's current position from odometry"""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        _, _, self.theta = euler_from_quaternion([
            orientation_q.x, orientation_q.y, 
            orientation_q.z, orientation_q.w
        ])
    
    def lidar_callback(self, msg):
        """Process LiDAR scan data"""
        if not self.lidar_received:
            self.get_logger().info("✓ LiDAR data CONNECTED!")
            self.get_logger().info(f"  Scan range: {msg.angle_min:.2f} to {msg.angle_max:.2f} rad")
            self.get_logger().info(f"  Number of readings: {len(msg.ranges)}")
            self.lidar_received = True
        
        self.lidar_ranges = np.array(msg.ranges)
        self.lidar_ranges[np.isinf(self.lidar_ranges)] = msg.range_max
        self.lidar_ranges[np.isnan(self.lidar_ranges)] = msg.range_max
        
        self.lidar_angle_min = msg.angle_min
        self.lidar_angle_max = msg.angle_max
        self.lidar_angle_increment = msg.angle_increment
    
    def filter_waypoints(self, waypoints):
        """
        Filter waypoints using Euclidean distance.
        Ignores consecutive waypoints closer than 1.0 unit.
        Always keeps first and last waypoint.
        """
        if len(waypoints) < 2:
            return waypoints
        
        filtered = [waypoints[0]]  # Always keep first waypoint
        
        for i in range(1, len(waypoints)):
            prev_x, prev_y = filtered[-1]
            curr_x, curr_y = waypoints[i]
            
            # Calculate Euclidean distance
            euclidean_distance = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            
            # Only add if distance is >= 1.0 unit
            if euclidean_distance >= self.waypoint_distance_threshold:
                filtered.append(waypoints[i])
        
        # Always keep last waypoint if it was filtered out
        if filtered[-1] != waypoints[-1]:
            filtered.append(waypoints[-1])
        
        original_count = len(waypoints)
        filtered_count = len(filtered)
        self.get_logger().info(
            f"Waypoint filtering: {original_count} → {filtered_count} "
            f"({original_count - filtered_count} removed, min distance: {self.waypoint_distance_threshold}m)"
        )
        
        return filtered
    
    def waypoint_callback(self, msg):
        """Parse waypoints from JSON path"""
        try:
            data = msg.data.strip()
            json_data = json.loads(data)
            
            if isinstance(json_data, dict) and 'path' in json_data:
                path = json_data['path']
                if isinstance(path, list):
                    raw_waypoints = []
                    for point in path:
                        if isinstance(point, list) and len(point) >= 2:
                            x, y = float(point[0]), float(point[1])
                            raw_waypoints.append((x, y))
                    
                    self.waypoints = self.filter_waypoints(raw_waypoints)
                    self.get_logger().info(
                        f"✓ Loaded waypoints from /maze/path_data"
                    )
        except Exception as e:
            self.get_logger().error(f"Error parsing waypoint: {e}")
    
    def get_distance(self, x_goal, y_goal):
        """Calculate distance to goal"""
        return math.sqrt((x_goal - self.x)**2 + (y_goal - self.y)**2)
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def get_caster_front_angle(self, x_goal, y_goal):
        """Calculate angle for caster to point towards goal"""
        angle_to_goal = math.atan2(y_goal - self.y, x_goal - self.x)
        caster_angle = self.normalize_angle(angle_to_goal + math.pi)
        return caster_angle
    
    def analyze_lidar_sectors(self):
        """
        Analyze LiDAR data in sectors to find best direction.
        Returns: (front_clear, best_direction, min_front_distance)
        """
        if len(self.lidar_ranges) == 0:
            return True, 0.0, float('inf')
        
        n = len(self.lidar_ranges)
        
        # Define sectors (relative to robot's current orientation)
        # Front: center ±30°, Left: -30° to -150°, Right: +30° to +150°
        front_indices = []
        left_indices = []
        right_indices = []
        
        for i in range(n):
            angle = self.lidar_angle_min + i * self.lidar_angle_increment
            # Normalize to robot frame
            if abs(angle) < self.front_angle_range / 2:
                front_indices.append(i)
            elif angle > 0:
                left_indices.append(i)
            else:
                right_indices.append(i)
        
        # Calculate minimum distances in each sector
        front_dist = np.min(self.lidar_ranges[front_indices]) if front_indices else float('inf')
        left_dist = np.mean(self.lidar_ranges[left_indices]) if left_indices else float('inf')
        right_dist = np.mean(self.lidar_ranges[right_indices]) if right_indices else float('inf')
        
        # Determine if front is clear
        front_clear = front_dist > self.danger_distance
        
        # Choose best direction (prefer larger clearance)
        if left_dist > right_dist:
            best_direction = 1.0  # Turn left
        else:
            best_direction = -1.0  # Turn right
        
        return front_clear, best_direction, front_dist
    
    def get_avoidance_velocity(self, x_goal, y_goal):
        """
        Enhanced obstacle avoidance with dynamic behavior.
        Returns: (linear_vel, angular_vel, should_stop)
        """
        front_clear, best_direction, min_distance = self.analyze_lidar_sectors()
        
        desired_angle = self.get_caster_front_angle(x_goal, y_goal)
        angle_error = self.normalize_angle(desired_angle - self.theta)
        
        # CRITICAL: Emergency stop
        if min_distance < self.critical_distance:
            self.stuck_counter += 1
            if self.stuck_counter > self.max_stuck_count:
                self.recovery_mode = True
                self.get_logger().warn("🔄 RECOVERY MODE: Executing escape maneuver")
            
            return 0.0, self.angular_speed * best_direction * 1.5, True
        
        # Reset stuck counter if we have clearance
        if min_distance > self.safe_distance:
            self.stuck_counter = 0
            self.recovery_mode = False
        
        # RECOVERY MODE: Aggressive turning and backup
        if self.recovery_mode:
            # Occasionally try backing up to escape
            if self.stuck_counter % 10 < 3:  # Back up for 3 out of every 10 cycles
                return 0.3, self.angular_speed * best_direction, False
            else:
                return 0.0, self.angular_speed * best_direction * 2.0, False
        
        # DANGER ZONE: Slow down and prepare to avoid
        if min_distance < self.danger_distance:
            self.avoiding_obstacle = True
            self.avoidance_direction = best_direction
            
            # Blend between goal-seeking and obstacle avoidance
            avoidance_weight = 1.0 - (min_distance - self.critical_distance) / (self.danger_distance - self.critical_distance)
            
            linear_vel = -self.linear_speed * 0.3 * (1.0 - avoidance_weight)
            angular_vel = (self.angular_speed * avoidance_weight * best_direction + 
                          2.0 * angle_error * (1.0 - avoidance_weight))
            
            return linear_vel, angular_vel, False
        
        # CAUTION ZONE: Reduce speed but continue
        if min_distance < self.safe_distance:
            speed_factor = (min_distance - self.danger_distance) / (self.safe_distance - self.danger_distance)
            linear_vel = -self.linear_speed * speed_factor * 0.7
            angular_vel = 2.0 * angle_error
            return linear_vel, angular_vel, False
        
        # CLEAR PATH: Normal navigation
        self.avoiding_obstacle = False
        return None, None, False
    
    def move_to_waypoint(self, x_goal, y_goal):
        """Navigate with enhanced obstacle avoidance"""
        twist = Twist()
        distance = self.get_distance(x_goal, y_goal)
        
        # Reached waypoint
        if distance < self.position_tolerance:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            self.stuck_counter = 0
            self.recovery_mode = False
            return True
        
        # Get avoidance behavior
        avoid_linear, avoid_angular, emergency_stop = self.get_avoidance_velocity(x_goal, y_goal)
        
        # Use avoidance behavior if needed
        if avoid_linear is not None:
            twist.linear.x = avoid_linear
            twist.angular.z = avoid_angular
        else:
            # Normal navigation
            desired_angle = self.get_caster_front_angle(x_goal, y_goal)
            angle_error = self.normalize_angle(desired_angle - self.theta)
            
            # Rotate first if not aligned
            if abs(angle_error) > self.angle_tolerance:
                twist.linear.x = 0.0
                twist.angular.z = self.angular_speed if angle_error > 0 else -self.angular_speed
            else:
                # Move with caster leading
                speed = min(self.linear_speed, distance)
                twist.linear.x = -speed
                twist.angular.z = 2.0 * angle_error
        
        self.cmd_vel_pub.publish(twist)
        return False
    
    def navigation_loop(self):
        """Main navigation loop"""
        if self.waypoints and not self.navigating:
            self.navigating = True
            self.current_waypoint = self.waypoints[0]
            x_goal, y_goal = self.current_waypoint
            self.get_logger().info(
                f"→ Waypoint {len(self.waypoints)}: ({x_goal:.2f}, {y_goal:.2f})"
            )
        
        if self.navigating and self.current_waypoint:
            x_goal, y_goal = self.current_waypoint
            if self.move_to_waypoint(x_goal, y_goal):
                self.waypoints.pop(0)
                self.current_waypoint = None
                self.navigating = False
                
                if len(self.waypoints) > 0:
                    self.get_logger().info(f"✓ Reached! {len(self.waypoints)} remaining")
                else:
                    self.get_logger().info("🎯 ALL WAYPOINTS COMPLETE!")

def main(args=None):
    rclpy.init(args=args)
    navigator = WaypointNavigator()
    
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()