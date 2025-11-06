#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        
        self.bridge = CvBridge()
        
        # Subscribe to camera
        self.camera_sub = self.create_subscription(
            Image,
            '/iris/downward_camera/image_raw',
            self.camera_callback,
            10
        )
        
        # Create window
        cv2.namedWindow('Drone Camera View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Drone Camera View', 800, 600)
        
        self.frame_count = 0
        self.get_logger().info('Camera Viewer Started')
        self.get_logger().info('Press "q" to quit')
        self.get_logger().info('Press "s" to save current frame')
    
    def camera_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            self.frame_count += 1
            
            # Add frame counter
            display_img = cv_image.copy()
            cv2.putText(display_img, f'Frame: {self.frame_count}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add timestamp
            cv2.putText(display_img, f'ROS Time: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show image
            cv2.imshow('Drone Camera View', display_img)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Quit requested')
                rclpy.shutdown()
            elif key == ord('s'):
                filename = f'/home/ps_9204/op/camera_frame_{self.frame_count}.png'
                cv2.imwrite(filename, cv_image)
                self.get_logger().info(f'Saved frame to {filename}')
                
        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    viewer = CameraViewer()
    
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        viewer.get_logger().info('Shutting down...')
    finally:
        cv2.destroyAllWindows()
        viewer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()