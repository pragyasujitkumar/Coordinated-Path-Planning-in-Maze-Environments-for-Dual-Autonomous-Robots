#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from heapq import heappush, heappop
import time
from std_msgs.msg import String
import json

GRID_SIZE = 100
PATH_THRESHOLD = 0.5
PATH_CLEARANCE = 3  # Increase this for more wall clearance

class MazeSolver(Node):
    def __init__(self):
        super().__init__('maze_solver')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, '/iris/downward_camera/image_raw', self.camera_callback, 10
        )

        self.path_pub = self.create_publisher(String, '/maze/path_data', 10)

        self.current_image = None
        self.maze_bbox = None
        self.maze_map = None
        self.start_pos = None
        self.end_pos = None
        self.path = None
        self.processing = False
        self.window_name = 'Maze Solver'

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_click)

        self.get_logger().info("Press 'p' to process maze, 'r' to reset, 'q' to quit.")

    def camera_callback(self, msg):
        self.current_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.display_image()

    def mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.maze_bbox is not None:
            x0, y0, w, h = self.maze_bbox
            if x0 <= x <= x0 + w and y0 <= y <= y0 + h:
                rel_x = (x - x0) / w
                rel_y = (y - y0) / h
                grid_x, grid_y = int(rel_y * GRID_SIZE), int(rel_x * GRID_SIZE)
                if self.start_pos is None:
                    self.start_pos = (grid_x, grid_y)
                    self.get_logger().info(f"Start point set at {self.start_pos}")
                elif self.end_pos is None:
                    self.end_pos = (grid_x, grid_y)
                    self.get_logger().info(f"End point set at {self.end_pos}")
                    self.solve_maze()

    def display_image(self):
        if self.current_image is None:
            return

        img = self.current_image.copy()
        if self.maze_bbox is not None:
            x, y, w, h = self.maze_bbox
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        if self.path is not None and self.maze_bbox is not None:
            self.draw_path(img)
        if self.start_pos and self.maze_bbox:
            self.draw_marker(img, self.start_pos, (0, 0, 255))
        if self.end_pos and self.maze_bbox:
            self.draw_marker(img, self.end_pos, (255, 0, 0))

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

    def draw_marker(self, img, grid_pos, color):
        if not self.maze_bbox: return
        x, y, w, h = self.maze_bbox
        px = int((grid_pos[1] / GRID_SIZE) * w + x)
        py = int((grid_pos[0] / GRID_SIZE) * h + y)
        cv2.circle(img, (px, py), 6, color, -1)


    def process_maze(self):
        if self.current_image is None:
            self.get_logger().warn("No image available.")
            return

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.get_logger().error("No maze found.")
            return

        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        self.maze_bbox = (x, y, w, h)
        maze_crop = binary[y:y+h, x:x+w]

        maze_crop = cv2.resize(maze_crop, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_NEAREST)

        # Erode walls inward to increase safe distance from walls
        kernel = np.ones((PATH_CLEARANCE, PATH_CLEARANCE), np.uint8)
        safe_maze = cv2.erode(maze_crop, kernel, iterations=1)

        # 1 = path, 0 = wall
        self.maze_map = (safe_maze > 127).astype(np.uint8)
        self.get_logger().info("Maze processed with safe clearance. Click to select START and END points.")

    def solve_maze(self):
        if self.start_pos is None or self.end_pos is None:
            self.get_logger().error("Both start and end points must be selected.")
            return

        self.get_logger().info("Running A* pathfinding...")
        path = self.a_star_search(self.start_pos, self.end_pos)
        
        if path:
            self.path = path
            self.get_logger().info(f"Path found: {len(path)} steps.")
            self.world_coords = []
            
            for r, c in self.path:
                # Convert grid coordinates to world coordinates
                # Grid: (0,0) = top-left, (GRID_SIZE-1, GRID_SIZE-1) = bottom-right
                # World: x ∈ [-6, 7], y ∈ [-5.5, 9.3]
                x_min, x_max = -6.0, 7.0
                y_min, y_max = -5.0, 9.3
                
                # Normalize to [0, 1]
                norm_col = c / (GRID_SIZE - 1)  # 0 = left edge, 1 = right edge
                norm_row = r / (GRID_SIZE - 1)  # 0 = top edge, 1 = bottom edge
                
                # Map to world coordinates
                # X-axis: left (-5) to right (+5)
                world_x = x_max - norm_row * (x_max - x_min)
                
                # Y-axis: top (+5) to bottom (-5) - inverted because grid row 0 is top
                world_y = y_max - norm_col * (y_max - y_min)

                self.world_coords.append((world_x, world_y))
                
            self.get_logger().info(f"World coordinates calculated: {len(self.world_coords)} waypoints")
            self.get_logger().info(f"First waypoint: {self.world_coords[0]}")
            self.get_logger().info(f"Last waypoint: {self.world_coords[-1]}")
    
            # self.get_logger().info(f"Path coordinates (world): {self.world_coords}")
            self.publish_path_data()
        else:
            self.get_logger().warn("No path found.")

    def a_star_search(self, start, goal):
        rows, cols = self.maze_map.shape

        # Use distance transform to bias towards center
        dist_from_wall = cv2.distanceTransform(self.maze_map.astype(np.uint8), cv2.DIST_L2, 3)
        dist_norm = dist_from_wall / dist_from_wall.max()

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
                    # Encourage path through central areas
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
            cv2.line(img, p1, p2, (0,255,0), 2)

    def publish_path_data(self):
            """Publish path data for coordinate conversion"""
            if self.path is None or self.world_coords is None or self.start_pos is None or self.end_pos is None:
                return
            
            data = {
                'path': self.world_coords,
                'start': list(self.start_pos),
                'end': list(self.end_pos),
                'maze_bbox': list(self.maze_bbox) if self.maze_bbox else None,
                'grid_size': GRID_SIZE
            }
            
            msg = String()
            msg.data = json.dumps(data)
            self.path_pub.publish(msg)
            self.get_logger().info("Published path data to /maze/path_data")
            
def main(args=None):
    rclpy.init(args=args)
    node = MazeSolver()
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
