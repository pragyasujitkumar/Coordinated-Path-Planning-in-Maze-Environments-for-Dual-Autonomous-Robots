#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
from heapq import heappush, heappop
from std_msgs.msg import String
import json
import math
import numpy as np

GRID_SIZE = 100
PATH_CLEARANCE = 3

class AdaptiveMazeSolver(Node):
    def __init__(self):
        super().__init__('adaptive_maze_solver')
        self.bridge = CvBridge()
        
        # Subscriptions for image
        self.image_sub = self.create_subscription(
            Image, '/iris/downward_camera/image_raw', self.camera_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/iris/downward_camera/camera_info', self.camera_info_callback, 10
        )
        
        # Try multiple pose topic sources
        self.pose_sub1 = self.create_subscription(
            PoseStamped, '/iris/pose', self.pose_callback, 10
        )
        self.pose_sub2 = self.create_subscription(
            Odometry, '/iris/odometry', self.odom_callback, 10
        )
        self.pose_sub3 = self.create_subscription(
            Odometry, '/iris/odom', self.odom_callback, 10
        )
        self.pose_sub4 = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        
        # Publisher
        self.path_pub = self.create_publisher(String, '/maze/path_data', 10)
        
        # Camera parameters
        self.camera_matrix = None
        self.drone_height = None
        self.drone_pose = None
        self.camera_calibrated = False
        self.pose_source = None
        
        # Manual height override (fallback)
        self.manual_height = 10.0  # Default drone height in meters
        self.use_manual_height = False
        
        # Camera/drone fixed offset (tunable)
        # If your drone/camera is offset +1.5m in world Y from origin, set drone_offset_y = -1.5
        # (sign depends on your world axis convention; try -1.5 first).
        self.drone_offset_x = 0.0
        self.drone_offset_y = -1.5   # <-- adjust sign/magnitude if needed
        self.drone_offset_z = 0.0
        
                # ---------- camera / transform tuning ----------
        # camera-to-body fixed rotation (roll,pitch,yaw) in radians.
        # For a downward facing camera, try pitch = -pi/2 (i.e. rotate -90 degrees around Y)
        # or pitch = +pi/2 depending on axis conventions. I suggest trying:
        # camera-to-body rotation (roll, pitch, yaw)
        # For a downward camera whose image x-axis runs left→right and y-axis top→bottom:
        # try one of these; we’ll start with yaw = π/2.
        self.cam_rpy = (0.0, -math.pi/2.0, math.pi/2.0)

        # camera fixed translation from body origin (in body frame), usually small (meters).
        self.cam_offset_body = (0.0, 0.0, 0.0)

        # ground plane z coordinate (maze floor). Change if maze is not at world z=0.
        self.ground_z = 0.0

        # Image and maze data
        self.current_image = None
        self.maze_bbox = None
        self.maze_map = None
        self.start_pos = None
        self.end_pos = None
        self.path = None
        
        # Real-world maze bounds
        self.maze_world_bounds = None
        
        # Window setup
        self.window_name = 'Adaptive Maze Solver'
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_click)
        
        self.get_logger().info("="*60)
        self.get_logger().info("ADAPTIVE MAZE SOLVER - Works with ANY maze!")
        self.get_logger().info("="*60)
        self.get_logger().info("Controls:")
        self.get_logger().info("  'p' - Process maze")
        self.get_logger().info("  'r' - Reset points")
        self.get_logger().info("  'h' - Toggle manual height mode")
        self.get_logger().info("  '+' - Increase manual height")
        self.get_logger().info("  '-' - Decrease manual height")
        self.get_logger().info("  'q' - Quit")
        self.get_logger().info("="*60)
        self.get_logger().info("Waiting for camera and drone position...")
        
        # Timer to check topic availability
        self.create_timer(5.0, self.check_topics)

    def check_topics(self):
        """Periodically check if we're receiving necessary data"""
        if not self.camera_calibrated:
            self.get_logger().warn("⚠ Camera info not received. Check /iris/downward_camera/camera_info topic")
        
        if self.drone_height is None and not self.use_manual_height:
            self.get_logger().warn("⚠ Drone pose not received. Press 'h' to use manual height mode")
            self.get_logger().info(f"  Trying topics: /iris/pose, /iris/odometry, /iris/odom, /odom")

    def camera_info_callback(self, msg):
        """Get camera intrinsic parameters"""
        if not self.camera_calibrated:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]
            self.camera_calibrated = True
            self.get_logger().info("✓ Camera calibrated")
            self.get_logger().info(f"  Focal length: fx={self.fx:.2f}, fy={self.fy:.2f}")
            self.get_logger().info(f"  Principal point: cx={self.cx:.2f}, cy={self.cy:.2f}")

    def pose_callback(self, msg):
        """Get drone position from PoseStamped"""
        if self.pose_source is None:
            self.pose_source = "PoseStamped"
            self.get_logger().info(f"✓ Pose data connected (PoseStamped)")
        
        self.drone_pose = msg.pose
        self.drone_height = msg.pose.position.z
        
        if not hasattr(self, '_height_logged'):
            self.get_logger().info(f"✓ Drone height: {self.drone_height:.2f}m")
            self.get_logger().info(f"✓ Drone position: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})")
            self._height_logged = True

    def odom_callback(self, msg):
        """Get drone position from Odometry"""
        if self.pose_source is None:
            self.pose_source = "Odometry"
            self.get_logger().info(f"✓ Pose data connected (Odometry)")
        
        self.drone_pose = msg.pose.pose
        self.drone_height = msg.pose.pose.position.z
        
        if not hasattr(self, '_height_logged'):
            self.get_logger().info(f"✓ Drone height: {self.drone_height:.2f}m")
            self.get_logger().info(f"✓ Drone position: ({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f}, {msg.pose.pose.position.z:.2f})")
            self._height_logged = True

    def pixel_to_world(self, pixel_x, pixel_y):
        """
        Accurate pixel -> world by forming a camera-ray and intersecting with ground plane z = ground_z.
        Returns (world_x, world_y) or (None, None) on failure.

        This version logs internal values for debugging.
        """
        height = self.manual_height if self.use_manual_height else self.drone_height
        if height is None or not self.camera_calibrated:
            return None, None

        if self.drone_pose is None:
            base_pos = np.array([0.0, 0.0, 0.0])
            base_q = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            base_pos = np.array([
                float(self.drone_pose.position.x),
                float(self.drone_pose.position.y),
                float(self.drone_pose.position.z)
            ])
            base_q = np.array([
                float(self.drone_pose.orientation.x),
                float(self.drone_pose.orientation.y),
                float(self.drone_pose.orientation.z),
                float(self.drone_pose.orientation.w)
            ])

        # intrinsics -> ray in camera frame
        x_cam = (pixel_x - self.cx) / self.fx
        y_cam = (pixel_y - self.cy) / self.fy
        ray_cam = np.array([x_cam, y_cam, 1.0], dtype=float)
        ray_cam = ray_cam / np.linalg.norm(ray_cam)

        # camera->body rotation & body->world rotation
        cr, cp, cy = self.cam_rpy
        R_cam_body = self.rpy_to_rotmat(cr, cp, cy)
        R_body_world = self.quat_to_rotmat(base_q)

        ray_body = R_cam_body @ ray_cam
        ray_world = R_body_world @ ray_body

        cam_offset_body = np.array(self.cam_offset_body)
        cam_world_origin = base_pos + (R_body_world @ cam_offset_body)

        ground_z = getattr(self, 'ground_z', 0.0)

        dz = ray_world[2]
        # debug log small summary
        self.get_logger().debug(
            f"pixel_to_world: pix=({pixel_x:.1f},{pixel_y:.1f}) "
            f"x_cam={x_cam:.4f} y_cam={y_cam:.4f} ray_world_z={dz:.6f} "
            f"cam_origin_z={cam_world_origin[2]:.4f}"
        )

        if abs(dz) < 1e-8:
            self.get_logger().debug("pixel_to_world: ray parallel to plane (dz≈0).")
            return None, None

        t = (ground_z - cam_world_origin[2]) / dz
        self.get_logger().debug(f"pixel_to_world: t={t:.6f}")

        if t <= 0:
            self.get_logger().debug("pixel_to_world: intersection behind camera (t<=0).")
            return None, None

        world_pt = cam_world_origin + t * ray_world
        world_x, world_y = float(world_pt[0]) + self.drone_offset_x, float(world_pt[1]) + self.drone_offset_y

        self.get_logger().debug(f"pixel_to_world: result ({world_x:.3f}, {world_y:.3f})")
        return world_x, world_y



    # ---------- helper functions ----------
    @staticmethod
    def rpy_to_rotmat(roll, pitch, yaw):
        """Return 3x3 rotation matrix from roll, pitch, yaw (ZYX convention: R = Rz(yaw)*Ry(pitch)*Rx(roll))."""
        cr = math.cos(roll); sr = math.sin(roll)
        cp = math.cos(pitch); sp = math.sin(pitch)
        cy = math.cos(yaw); sy = math.sin(yaw)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ], dtype=float)
        return R

    @staticmethod
    def quat_to_rotmat(q):
        """Convert quaternion [x,y,z,w] to 3x3 rotation matrix (maps vector_from_body -> world)."""
        x, y, z, w = q
        # normalize
        n = math.sqrt(x*x + y*y + z*z + w*w)
        if n == 0:
            return np.eye(3)
        x /= n; y /= n; z /= n; w /= n
        xx = x*x; yy = y*y; zz = z*z
        xy = x*y; xz = x*z; yz = y*z
        xw = x*w; yw = y*w; zw = z*w
        R = np.array([
            [1 - 2*(yy+zz),     2*(xy - zw),       2*(xz + yw)],
            [2*(xy + zw),       1 - 2*(xx+zz),     2*(yz - xw)],
            [2*(xz - yw),       2*(yz + xw),       1 - 2*(xx+yy)]
        ], dtype=float)
        return R



    def calculate_maze_world_bounds(self):
        """Calculate real-world bounds of the maze. Tries flipping camera pitch sign if first attempt fails."""
        if self.maze_bbox is None:
            return False

        x, y, w, h = self.maze_bbox
        corners_pixel = [
            (x, y),
            (x + w, y),
            (x, y + h),
            (x + w, y + h)
        ]

        # Helper to project corners with current cam_rpy and count valid
        def project_corners():
            corners_world = []
            for px, py in corners_pixel:
                wx, wy = self.pixel_to_world(px, py)
                if wx is not None and wy is not None:
                    corners_world.append((wx, wy))
                else:
                    corners_world.append((None, None))
            valid_count = sum(1 for c in corners_world if c[0] is not None)
            return corners_world, valid_count

        # First try with current cam_rpy
        corners_world, valid_count = project_corners()
        self.get_logger().info(f"calculate_maze_world_bounds: valid corner projections = {valid_count}/4 using cam_rpy={self.cam_rpy}")

        # If not all corners valid, try flipping the pitch sign (common fix for downward cam)
        if valid_count < 4:
            orig = self.cam_rpy
            flipped = (orig[0], -orig[1], orig[2])
            self.get_logger().info(f"Trying flipped cam_rpy {flipped} to recover projections...")
            self.cam_rpy = flipped
            corners_world, valid_count = project_corners()
            self.get_logger().info(f"After flip: valid corner projections = {valid_count}/4 using cam_rpy={self.cam_rpy}")
            if valid_count < 4:
                # revert cam_rpy if flip didn't help
                self.get_logger().warn("Projection still incomplete after flip. Will abort world bounds calculation and print debug info.")
                self.cam_rpy = orig
                # print debug of each corner (for the user)
                for i, (px, py) in enumerate(corners_pixel):
                    wx, wy = self.pixel_to_world(px, py)
                    self.get_logger().error(f"DEBUG corner[{i}] pixel ({px:.1f},{py:.1f}) -> world ({wx},{wy})")
                return False
        else:
            # all good; ensure cam_rpy is what we used (it already is)
            pass

        # collect final valid corners (after potential flip)
        final_corners = []
        for (px, py) in corners_pixel:
            wx, wy = self.pixel_to_world(px, py)
            if wx is None or wy is None:
                # should not happen here, but be safe
                self.get_logger().error(f"Unexpected None for corner pixel ({px},{py})")
                return False
            final_corners.append((wx, wy))

        x_coords = [c[0] for c in final_corners]
        y_coords = [c[1] for c in final_corners]

        self.maze_world_bounds = {
            'x_min': min(x_coords),
            'x_max': max(x_coords),
            'y_min': min(y_coords),
            'y_max': max(y_coords)
        }

        # debug: project center
        cx_pix = x + w/2.0
        cy_pix = y + h/2.0
        wxc, wyc = self.pixel_to_world(cx_pix, cy_pix)
        self.get_logger().info(f"DEBUG: maze center pixel -> world: ({wxc}, {wyc})")

        width = self.maze_world_bounds['x_max'] - self.maze_world_bounds['x_min']
        height = self.maze_world_bounds['y_max'] - self.maze_world_bounds['y_min']

        self.get_logger().info("="*60)
        self.get_logger().info("✓ MAZE WORLD BOUNDS CALCULATED:")
        self.get_logger().info(f"  X: [{self.maze_world_bounds['x_min']:.2f}, {self.maze_world_bounds['x_max']:.2f}]")
        self.get_logger().info(f"  Y: [{self.maze_world_bounds['y_min']:.2f}, {self.maze_world_bounds['y_max']:.2f}]")
        self.get_logger().info(f"  Size: {width:.2f}m × {height:.2f}m")
        if self.use_manual_height:
            self.get_logger().info(f"  Using manual height: {self.manual_height:.2f}m")
        else:
            self.get_logger().info(f"  Using drone height: {self.drone_height:.2f}m")
        self.get_logger().info("="*60)

        return True


    def grid_to_world(self, grid_row, grid_col):
        """Convert grid coordinates to world coordinates"""
        if self.maze_world_bounds is None:
            return None, None
        
        # Normalize grid position to [0, 1]
        norm_col = grid_col / (GRID_SIZE - 1)
        norm_row = grid_row / (GRID_SIZE - 1)
        
        # Map to world coordinates
        world_x = self.maze_world_bounds['x_min'] + norm_col * (
            self.maze_world_bounds['x_max'] - self.maze_world_bounds['x_min']
        )
        world_y = self.maze_world_bounds['y_min'] + norm_row * (
            self.maze_world_bounds['y_max'] - self.maze_world_bounds['y_min']
        )
        
        return world_x, world_y

    def camera_callback(self, msg):
        self.current_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.display_image()

    def mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.maze_bbox is not None:
            x0, y0, w, h = self.maze_bbox
            if x0 <= x <= x0 + w and y0 <= y <= y0 + h:
                rel_x = (x - x0) / float(w)
                rel_y = (y - y0) / float(h)
                grid_row = int(rel_y * GRID_SIZE)
                grid_col = int(rel_x * GRID_SIZE)

                # clamp indices
                grid_row = max(0, min(GRID_SIZE-1, grid_row))
                grid_col = max(0, min(GRID_SIZE-1, grid_col))

                if self.start_pos is None:
                    self.start_pos = (grid_row, grid_col)
                    world_x, world_y = self.grid_to_world(grid_row, grid_col)
                    if world_x is None or world_y is None:
                        self.get_logger().error("START: projection to world failed (None). Check debug logs.")
                        self.start_pos = None
                    else:
                        self.get_logger().info(f"START: Grid({grid_row}, {grid_col}) → World({world_x:.2f}, {world_y:.2f})")
                elif self.end_pos is None:
                    self.end_pos = (grid_row, grid_col)
                    world_x, world_y = self.grid_to_world(grid_row, grid_col)
                    if world_x is None or world_y is None:
                        self.get_logger().error("END: projection to world failed (None). Check debug logs.")
                        self.end_pos = None
                    else:
                        self.get_logger().info(f"END: Grid({grid_row}, {grid_col}) → World({world_x:.2f}, {world_y:.2f})")
                        self.solve_maze()


    def display_image(self):
        if self.current_image is None:
            return

        img = self.current_image.copy()
        
        # Draw maze bounding box
        if self.maze_bbox is not None:
            x, y, w, h = self.maze_bbox
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # Display maze info
            info_y = 30
            if self.maze_world_bounds:
                world_w = self.maze_world_bounds['x_max'] - self.maze_world_bounds['x_min']
                world_h = self.maze_world_bounds['y_max'] - self.maze_world_bounds['y_min']
                info = f"Maze: {world_w:.1f}x{world_h:.1f}m"
                cv2.putText(img, info, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display height mode (safe formatting)
        if self.use_manual_height:
            height_val = self.manual_height
            height_text = f"Height: {height_val:.1f}m (MANUAL)"
            text_color = (0, 165, 255)
        else:
            if self.drone_height is None:
                height_text = "Height: N/A"
                text_color = (0, 165, 255)
            else:
                height_val = self.drone_height
                height_text = f"Height: {height_val:.1f}m"
                text_color = (0, 255, 0)

        cv2.putText(img, height_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Draw path
        if self.path is not None and self.maze_bbox is not None:
            self.draw_path(img)
        
        # Draw markers
        if self.start_pos and self.maze_bbox:
            self.draw_marker(img, self.start_pos, (0, 0, 255), "START")
        if self.end_pos and self.maze_bbox:
            self.draw_marker(img, self.end_pos, (255, 0, 0), "END")

        cv2.imshow(self.window_name, img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            self.get_logger().info("Quitting...")
            rclpy.shutdown()
        elif key == ord('p'):
            self.get_logger().info("Processing maze...")
            self.process_maze()
        elif key == ord('r'):
            self.get_logger().info("Resetting...")
            self.start_pos = self.end_pos = self.path = None
        elif key == ord('h'):
            self.use_manual_height = not self.use_manual_height
            mode = "MANUAL" if self.use_manual_height else "AUTO"
            # safe formatting when drone_height might be None
            if self.use_manual_height:
                self.get_logger().info(f"Height mode: {mode} ({self.manual_height:.1f}m)")
            else:
                dh = self.drone_height if self.drone_height is not None else float('nan')
                if not np.isnan(dh):
                    self.get_logger().info(f"Height mode: {mode} ({dh:.1f}m)")
                else:
                    self.get_logger().info(f"Height mode: {mode} (drone height N/A)")
        elif key == ord('+') or key == ord('='):
            self.manual_height += 0.5
            self.get_logger().info(f"Manual height: {self.manual_height:.1f}m")
        elif key == ord('-') or key == ord('_'):
            self.manual_height = max(1.0, self.manual_height - 0.5)
            self.get_logger().info(f"Manual height: {self.manual_height:.1f}m")

    def draw_marker(self, img, grid_pos, color, label=None):
        if not self.maze_bbox:
            return
        x, y, w, h = self.maze_bbox
        px = int((grid_pos[1] / GRID_SIZE) * w + x)
        py = int((grid_pos[0] / GRID_SIZE) * h + y)
        cv2.circle(img, (px, py), 8, color, -1)
        cv2.circle(img, (px, py), 10, (255, 255, 255), 2)
        
        if label:
            cv2.putText(img, label, (px + 15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def process_maze(self):
        if self.current_image is None:
            self.get_logger().warn("No image available.")
            return
        
        if not self.camera_calibrated:
            self.get_logger().error("Camera not calibrated!")
            return
        
        if self.drone_height is None and not self.use_manual_height:
            self.get_logger().error("Drone position unknown! Press 'h' to use manual height mode")
            return

        # Convert to binary
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find maze contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.get_logger().error("No maze found in image.")
            return

        # Get largest contour
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        self.maze_bbox = (x, y, w, h)
        
        # Calculate real-world bounds
        if not self.calculate_maze_world_bounds():
            self.get_logger().error("Failed to calculate world bounds")
            return
        
        # Crop and resize maze
        maze_crop = binary[y:y+h, x:x+w]
        maze_crop = cv2.resize(maze_crop, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_NEAREST)

        # Apply safety clearance
        kernel = np.ones((PATH_CLEARANCE, PATH_CLEARANCE), np.uint8)
        safe_maze = cv2.erode(maze_crop, kernel, iterations=1)

        self.maze_map = (safe_maze > 127).astype(np.uint8)
        
        self.get_logger().info("✓ Maze processed. Click START and END points.")

    def solve_maze(self):
        if self.start_pos is None or self.end_pos is None:
            self.get_logger().error("Both start and end points must be selected.")
            return

        self.get_logger().info("Running A* pathfinding...")
        path = self.a_star_search(self.start_pos, self.end_pos)
        
        if path:
            self.path = path
            self.get_logger().info(f"✓ Path found: {len(path)} steps")
            
            # Convert to world coordinates
            self.world_coords = []
            for r, c in self.path:
                world_x, world_y = self.grid_to_world(r, c)
                if world_x is not None and world_y is not None:
                    self.world_coords.append((world_x, world_y))
            
            self.get_logger().info(f"✓ Converted to {len(self.world_coords)} world waypoints")
            self.get_logger().info(f"  First: ({self.world_coords[0][0]:.2f}, {self.world_coords[0][1]:.2f})")
            self.get_logger().info(f"  Last: ({self.world_coords[-1][0]:.2f}, {self.world_coords[-1][1]:.2f})")
            
            self.publish_path_data()
        else:
            self.get_logger().warn("No path found!")

    def a_star_search(self, start, goal):
        """A* pathfinding with distance transform bias"""
        rows, cols = self.maze_map.shape

        dist_from_wall = cv2.distanceTransform(self.maze_map.astype(np.uint8), cv2.DIST_L2, 3)
        dist_norm = dist_from_wall / (dist_from_wall.max() + 1e-6)

        def heuristic(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        open_set = []
        heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
        visited = set()
        g_score = {start: 0}
        count = 0

        while open_set:
            _, _, current, path = heappop(open_set)
            if current == goal:
                return path
            if current in visited:
                continue
            visited.add(current)

            for d in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = current[0]+d[0], current[1]+d[1]
                if 0 <= nr < rows and 0 <= nc < cols and self.maze_map[nr,nc] == 1:
                    cost = 1.0 / (0.1 + dist_norm[nr, nc])
                    new_cost = g_score[current] + cost
                    if (nr,nc) not in g_score or new_cost < g_score[(nr,nc)]:
                        g_score[(nr,nc)] = new_cost
                        count += 1
                        f = new_cost + heuristic((nr,nc), goal)
                        heappush(open_set, (f, count, (nr,nc), path + [(nr,nc)]))
        return None

    def draw_path(self, img):
        if self.path is None or self.maze_bbox is None:
            return
        x, y, w, h = self.maze_bbox
        for i in range(len(self.path)-1):
            r1, c1 = self.path[i]
            r2, c2 = self.path[i+1]
            p1 = (int((c1 / GRID_SIZE) * w + x), int((r1 / GRID_SIZE) * h + y))
            p2 = (int((c2 / GRID_SIZE) * w + x), int((r2 / GRID_SIZE) * h + y))
            cv2.line(img, p1, p2, (0, 255, 0), 2)

    def publish_path_data(self):
        """Publish path with world coordinates"""
        if self.path is None or self.world_coords is None:
            return
        
        data = {
            'path': self.world_coords,
            'start': list(self.start_pos),
            'end': list(self.end_pos),
            'maze_bbox': list(self.maze_bbox) if self.maze_bbox else None,
            'maze_world_bounds': self.maze_world_bounds,
            'grid_size': GRID_SIZE,
            'drone_height': self.manual_height if self.use_manual_height else self.drone_height
        }
        
        msg = String()
        msg.data = json.dumps(data)
        self.path_pub.publish(msg)
        self.get_logger().info("✓ Published path data to /maze/path_data")

def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveMazeSolver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()