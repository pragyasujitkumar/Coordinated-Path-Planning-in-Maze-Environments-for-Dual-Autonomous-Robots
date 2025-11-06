#!/usr/bin/env python3

"""
eBot Autonomous Waypoint Navigation - Optimized Controller
Combines reactive VFH navigation with improved obstacle detection
Forward-only movement with smart wall avoidance
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from gazebo_msgs.srv import SpawnEntity
from tf_transformations import euler_from_quaternion
import math
import json
import os
import numpy as np

class EBotNavigator(Node):
    def __init__(self):
        super().__init__('ebot_nav_task1A')
        # Set absolute path directly
        self.urdf_path = '/home/ps_9204/try_rppm_ws/src/maze_solver/urdf/bot.urdf'

        # You can still keep these as parameters if needed
        self.declare_parameter('robot_name', 'bot')
        self.declare_parameter('robot_namespace', '')

        self.robot_name = self.get_parameter('robot_name').value
        self.robot_namespace = self.get_parameter('robot_namespace').value
        # Publishers and Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Multiple LiDAR topic subscriptions
        self.lidar_sub = self.create_subscription(LaserScan, '/gazebo_lidar/out', self.lidar_callback, 10)
        self.lidar_sub2 = self.create_subscription(LaserScan, '/laser_scan', self.lidar_callback, 10)
        self.lidar_sub3 = self.create_subscription(LaserScan, '/lidar', self.lidar_callback, 10)
        
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.path_sub = self.create_subscription(String, '/maze/path_data', self.path_callback, 10)
        
        # Service client for spawning
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        
        # Waypoints (will be loaded from topic)
        self.waypoints = []
        self.waypoints_loaded = False
        self.robot_spawned = False
        
        # Waypoint filtering threshold
        self.waypoint_distance_threshold = 2.0
        
        # Tolerances
        self.position_tolerance = 0.25
        self.orientation_tolerance = math.radians(15)
        
        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.odom_received = False
        
        # LiDAR data
        self.scan_ranges = []
        self.scan_angle_min = 0.0
        self.scan_angle_increment = 0.0
        self.num_readings = 0
        
        # Enhanced obstacle avoidance parameters
        self.safe_distance = 0.5      # Start adjusting path
        self.danger_distance = 0.35   # Active avoidance
        self.critical_distance = 0.25 # Emergency stop
        self.wall_follow_distance = 0.4  # Preferred distance from walls
        
        # Control parameters (FORWARD ONLY)
        self.max_linear_speed = 0.4
        self.min_linear_speed = 0.1
        self.max_angular_speed = 1.2
        self.kp_linear = 0.6
        self.kp_angular = 1.5
        
        # Navigation state
        self.current_waypoint_idx = 0
        self.navigation_complete = False
        self.at_position = False
        
        # Stuck detection
        self.stuck_counter = 0
        self.max_stuck_count = 30
        self.last_position = (0.0, 0.0)
        self.position_check_timer = 0
        
        # Timer for control loop (20Hz for smoother control)
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('=== eBot Optimized Navigator Initialized ===')
        self.get_logger().info('Forward-only movement with reactive obstacle avoidance')
        self.get_logger().info('Waiting for path data from /maze/path_data...')
    
    def path_callback(self, msg):
        """Parse waypoints from JSON string and spawn robot at first waypoint"""
        if self.waypoints_loaded:
            return
        
        try:
            path_data = json.loads(msg.data)
            
            if not isinstance(path_data, dict):
                self.get_logger().error('Invalid path data format!')
                return
            
            if "path" in path_data:
                waypoints_raw = path_data["path"]
            elif "waypoints" in path_data:
                waypoints_raw = path_data["waypoints"]
            else:
                self.get_logger().error('No "path" or "waypoints" key found!')
                return
            
            if not isinstance(waypoints_raw, list):
                self.get_logger().error('Path data is not a list!')
                return
            
            # Convert to internal format [x, y, yaw]
            raw_waypoints = []
            for wp in waypoints_raw:
                if isinstance(wp, list) and len(wp) >= 2:
                    x = float(wp[0])
                    y = float(wp[1])
                    yaw = float(wp[2]) if len(wp) >= 3 else 0.0
                    raw_waypoints.append([x, y, yaw])
            
            if not raw_waypoints:
                self.get_logger().error('No valid waypoints found!')
                return
            
            # Filter waypoints
            self.waypoints = self.filter_waypoints(raw_waypoints)
            
            if self.waypoints:
                self.waypoints_loaded = True
                self.get_logger().info(f'✓ Filtered {len(raw_waypoints)} → {len(self.waypoints)} waypoints')
                
                # Spawn robot at first waypoint
                first_x, first_y, first_yaw = self.waypoints[0]
                self.spawn_robot(first_x, first_y, first_yaw)
            else:
                self.get_logger().error('No waypoints remain after filtering!')
                
        except Exception as e:
            self.get_logger().error(f'Error loading waypoints: {e}')
    
    def filter_waypoints(self, waypoints):
        """Filter waypoints to remove those too close together"""
        if not waypoints:
            return []
        
        filtered = [waypoints[0]]
        
        for i in range(1, len(waypoints)):
            prev_x, prev_y, _ = filtered[-1]
            curr_x, curr_y, curr_yaw = waypoints[i]
            
            distance = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            
            if distance >= self.waypoint_distance_threshold:
                filtered.append([curr_x, curr_y, curr_yaw])
        
        # Always keep last waypoint
        if len(waypoints) > 1 and waypoints[-1] not in filtered:
            filtered.append(waypoints[-1])
        
        return filtered
    
    def spawn_robot(self, x, y, yaw):
        """Spawn URDF robot at specified position"""
        self.get_logger().info('Waiting for /spawn_entity service...')
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            pass
        
        try:
            if self.urdf_path and os.path.exists(self.urdf_path):
                with open(self.urdf_path, 'r') as f:
                    urdf_content = f.read()
                self.get_logger().info(f'Loaded URDF: {self.urdf_path}')
            else:
                urdf_content = self.get_default_urdf()
                self.get_logger().info('Using default URDF')
        except Exception as e:
            self.get_logger().error(f'URDF error: {e}')
            urdf_content = self.get_default_urdf()
        
        request = SpawnEntity.Request()
        request.name = self.robot_name
        request.xml = urdf_content
        request.robot_namespace = self.robot_namespace
        
        initial_pose = Pose()
        initial_pose.position.x = x
        initial_pose.position.y = y
        initial_pose.position.z = 0.1
        initial_pose.orientation.z = math.sin(yaw / 2.0)
        initial_pose.orientation.w = math.cos(yaw / 2.0)
        request.initial_pose = initial_pose
        
        self.get_logger().info(f'Spawning at ({x:.2f}, {y:.2f}, {yaw:.2f})...')
        future = self.spawn_client.call_async(request)
        future.add_done_callback(self.spawn_callback)
    
    def spawn_callback(self, future):
        """Handle spawn response"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('✓ Robot spawned successfully!')
                self.robot_spawned = True
                self.last_position = (self.current_x, self.current_y)
                self.get_logger().info('\n=== Starting Navigation ===')
        except Exception as e:
            self.get_logger().error(f'Spawn failed: {e}')
    
    def get_default_urdf(self):
        """Minimal URDF"""
        return """<?xml version="1.0"?>
<robot name="ebot">
  <link name="base_link">
    <visual>
      <geometry><box size="0.4 0.3 0.2"/></geometry>
      <material name="blue"><color rgba="0 0 0.8 1"/></material>
    </visual>
    <collision>
      <geometry><box size="0.4 0.3 0.2"/></geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
</robot>"""
    
    def odom_callback(self, msg):
        """Update robot position"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.current_yaw = euler_from_quaternion(orientation_list)
        
        self.odom_received = True
    
    def lidar_callback(self, msg):
        """Process LiDAR data"""
        self.scan_ranges = np.array(msg.ranges)
        # Replace inf/nan with max range
        self.scan_ranges[np.isinf(self.scan_ranges)] = msg.range_max
        self.scan_ranges[np.isnan(self.scan_ranges)] = msg.range_max
        
        self.scan_angle_min = msg.angle_min
        self.scan_angle_increment = msg.angle_increment
        self.num_readings = len(self.scan_ranges)
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def get_distance_to_goal(self, goal_x, goal_y):
        """Calculate distance to goal"""
        dx = goal_x - self.current_x
        dy = goal_y - self.current_y
        return math.sqrt(dx**2 + dy**2)
    
    def get_angle_to_goal(self, goal_x, goal_y):
        """Calculate angle to goal"""
        dx = goal_x - self.current_x
        dy = goal_y - self.current_y
        angle_to_goal = math.atan2(dy, dx)
        return self.normalize_angle(angle_to_goal - self.current_yaw)
    
    def analyze_obstacles(self, goal_angle):
        """
        Enhanced obstacle analysis with sector-based approach
        Returns: (can_move_forward, best_direction, min_front_dist, clearance_score)
        """
        if len(self.scan_ranges) == 0:
            return True, goal_angle, float('inf'), 1.0
        
        # Analyze front sector (±45°)
        front_angle_range = math.pi / 4
        front_indices = []
        left_clearance = []
        right_clearance = []
        
        for i in range(self.num_readings):
            angle = self.scan_angle_min + i * self.scan_angle_increment
            angle_normalized = self.normalize_angle(angle)
            distance = self.scan_ranges[i]
            
            # Front sector
            if abs(angle_normalized) < front_angle_range:
                front_indices.append(i)
            # Left side
            elif 0 < angle_normalized < math.pi / 2:
                left_clearance.append(distance)
            # Right side
            elif -math.pi / 2 < angle_normalized < 0:
                right_clearance.append(distance)
        
        # Get minimum front distance
        if front_indices:
            min_front_dist = np.min(self.scan_ranges[front_indices])
        else:
            min_front_dist = float('inf')
        
        # Calculate side clearances
        avg_left = np.mean(left_clearance) if left_clearance else float('inf')
        avg_right = np.mean(right_clearance) if right_clearance else float('inf')
        
        # Determine if front is blocked
        can_move_forward = min_front_dist > self.danger_distance
        
        # Find best direction using Vector Field Histogram
        num_sectors = 36
        sector_size = 2 * math.pi / num_sectors
        sector_costs = np.zeros(num_sectors)
        
        for i, distance in enumerate(self.scan_ranges):
            if distance > 0.1 and distance < 10.0:
                angle = self.scan_angle_min + i * self.scan_angle_increment
                angle = self.normalize_angle(angle)
                
                sector_idx = int((angle + math.pi) / sector_size) % num_sectors
                
                # Higher cost for closer obstacles
                if distance < self.safe_distance:
                    cost = (self.safe_distance - distance) / self.safe_distance
                    sector_costs[sector_idx] += cost * 3.0
        
        # Find best sector considering goal direction
        best_sector = -1
        best_cost = float('inf')
        
        for sector_idx in range(num_sectors):
            sector_angle = -math.pi + sector_idx * sector_size + sector_size / 2
            sector_angle = self.normalize_angle(sector_angle)
            
            # Cost = obstacle_cost + goal_deviation
            angle_diff = abs(self.normalize_angle(sector_angle - goal_angle))
            goal_cost = angle_diff / math.pi
            
            total_cost = sector_costs[sector_idx] + goal_cost * 2.0
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_sector = sector_idx
        
        best_angle = -math.pi + best_sector * sector_size + sector_size / 2
        best_direction = self.normalize_angle(best_angle)
        
        # Calculate clearance score (0-1)
        clearance_score = min(min_front_dist / self.safe_distance, 1.0)
        
        return can_move_forward, best_direction, min_front_dist, clearance_score
    
    def check_if_stuck(self):
        """Detect if robot is stuck"""
        self.position_check_timer += 1
        
        if self.position_check_timer >= 20:  # Check every second
            dx = self.current_x - self.last_position[0]
            dy = self.current_y - self.last_position[1]
            distance_moved = math.sqrt(dx**2 + dy**2)
            
            if distance_moved < 0.05:  # Moved less than 5cm
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            
            self.last_position = (self.current_x, self.current_y)
            self.position_check_timer = 0
            
            return self.stuck_counter > self.max_stuck_count
        
        return False
    
    def control_loop(self):
        """Main control loop"""
        
        if not self.robot_spawned or not self.waypoints_loaded:
            return
        
        if not self.odom_received or len(self.scan_ranges) == 0:
            return
        
        # Check completion
        if self.navigation_complete or self.current_waypoint_idx >= len(self.waypoints):
            if not self.navigation_complete:
                self.get_logger().info('🎯 ALL WAYPOINTS COMPLETE!')
                self.navigation_complete = True
            self.stop_robot()
            return
        
        # Get current goal
        goal_x, goal_y, goal_yaw = self.waypoints[self.current_waypoint_idx]
        
        # Calculate errors
        distance_to_goal = self.get_distance_to_goal(goal_x, goal_y)
        angle_to_goal = self.get_angle_to_goal(goal_x, goal_y)
        orientation_error = self.normalize_angle(goal_yaw - self.current_yaw)
        
        cmd = Twist()
        
        # Check if reached position
        if distance_to_goal < self.position_tolerance:
            if not self.at_position:
                self.at_position = True
                self.get_logger().info(f'✓ Position {self.current_waypoint_idx + 1} reached')
            
            # Adjust final orientation
            if abs(orientation_error) < self.orientation_tolerance:
                self.get_logger().info(f'✓ Waypoint {self.current_waypoint_idx + 1} complete!')
                self.current_waypoint_idx += 1
                self.at_position = False
                self.stuck_counter = 0
                
                if self.current_waypoint_idx < len(self.waypoints):
                    self.get_logger().info(f'→ Next: Waypoint {self.current_waypoint_idx + 1}')
            else:
                # Rotate to final orientation
                cmd.angular.z = self.kp_angular * orientation_error
                cmd.angular.z = max(min(cmd.angular.z, self.max_angular_speed), -self.max_angular_speed)
        
        else:
            # FORWARD NAVIGATION WITH OBSTACLE AVOIDANCE
            
            # Analyze obstacles
            can_move, best_direction, min_dist, clearance = self.analyze_obstacles(angle_to_goal)
            
            # Check if stuck
            if self.check_if_stuck():
                self.get_logger().warn('⚠ Stuck detected! Aggressive turn...')
                cmd.linear.x = 0.0
                cmd.angular.z = self.max_angular_speed * (1.0 if best_direction > 0 else -1.0)
                self.stuck_counter = 0  # Reset after maneuver
            
            # CRITICAL: Very close obstacle
            elif min_dist < self.critical_distance:
                cmd.linear.x = 0.0
                cmd.angular.z = self.max_angular_speed * (1.0 if best_direction > 0 else -1.0)
                self.get_logger().warn(f'⚠ Critical distance: {min_dist:.2f}m')
            
            # DANGER: Close obstacle, turn more
            elif min_dist < self.danger_distance or not can_move:
                # Reduced forward speed, stronger turning
                cmd.linear.x = self.min_linear_speed * 0.5
                cmd.angular.z = self.kp_angular * 1.5 * best_direction
                cmd.angular.z = max(min(cmd.angular.z, self.max_angular_speed), -self.max_angular_speed)
            
            # CAUTION: Reduce speed near obstacles
            elif min_dist < self.safe_distance:
                speed_factor = clearance * 0.8
                cmd.linear.x = self.max_linear_speed * speed_factor
                cmd.linear.x = max(cmd.linear.x, self.min_linear_speed)
                
                # Blend goal seeking with obstacle avoidance
                steering = 0.6 * best_direction + 0.4 * angle_to_goal
                cmd.angular.z = self.kp_angular * steering
                cmd.angular.z = max(min(cmd.angular.z, self.max_angular_speed), -self.max_angular_speed)
            
            # CLEAR: Normal navigation
            else:
                # Proportional speed based on distance
                cmd.linear.x = min(self.max_linear_speed, self.kp_linear * distance_to_goal)
                cmd.linear.x = max(cmd.linear.x, self.min_linear_speed)
                
                # Smooth steering towards goal with slight avoidance bias
                steering = 0.8 * angle_to_goal + 0.2 * best_direction
                cmd.angular.z = self.kp_angular * steering
                cmd.angular.z = max(min(cmd.angular.z, self.max_angular_speed), -self.max_angular_speed)
        
        # Ensure forward only (no backward movement)
        cmd.linear.x = max(cmd.linear.x, 0.0)
        
        self.cmd_vel_pub.publish(cmd)
    
    def stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    navigator = EBotNavigator()
    
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        navigator.get_logger().info('Navigation interrupted')
    finally:
        navigator.stop_robot()
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()