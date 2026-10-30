#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np


class BoxDetector(Node):
    def __init__(self):
        super().__init__('box_detector')
        self.bridge = CvBridge()
        self.last_log_time = 0.0

        self.subscription = self.create_subscription(
            Image, '/camera', self.image_callback, 10
        )
        self.image_publisher = self.create_publisher(
            Image, '/camera/detection', 10
        )
        self.pixel_publisher = self.create_publisher(
            PointStamped, '/detected_box/pixel', 10
        )
        self.get_logger().info('Red box detector started')

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='bgr8'
        )
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(
            hsv, np.array([0, 80, 50]), np.array([10, 255, 255])
        )
        mask2 = cv2.inRange(
            hsv, np.array([170, 80, 50]), np.array([180, 255, 255])
        )
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        output = image.copy()
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / float(h)
            if 0.5 <= ratio <= 2.0:
                candidates.append((area, x, y, w, h))

        if candidates:
            area, x, y, w, h = max(candidates, key=lambda item: item[0])
            cx = x + w // 2
            cy = y + h // 2

            cv2.rectangle(
                output, (x, y), (x + w, y + h), (0, 255, 0), 2
            )
            cv2.circle(output, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(
                output, f'center=({cx},{cy}) area={int(area)}',
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1
            )

            point = PointStamped()
            point.header = msg.header
            point.point.x = float(cx)
            point.point.y = float(cy)
            point.point.z = float(area)
            self.pixel_publisher.publish(point)

            now = time.monotonic()
            if now - self.last_log_time >= 1.0:
                self.get_logger().info(
                    f'Red box center: pixel_x={cx}, pixel_y={cy}, '
                    f'area={int(area)}'
                )
                self.last_log_time = now

        out_msg = self.bridge.cv2_to_imgmsg(output, encoding='bgr8')
        out_msg.header = msg.header
        self.image_publisher.publish(out_msg)


def main():
    rclpy.init()
    node = BoxDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()