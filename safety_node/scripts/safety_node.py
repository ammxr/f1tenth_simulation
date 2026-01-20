#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class SafetyNode(Node):
    """
    The class that handles emergency braking.
    """
    def __init__(self):
        super().__init__('safety_node')
        """
        One publisher should publish to the /drive topic with a AckermannDriveStamped drive message.

        You should also subscribe to the /scan topic to get the LaserScan messages and
        the /ego_racecar/odom topic to get the current speed of the vehicle.

        The subscribers should use the provided odom_callback and scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        """
        self.speed = 0.
        # Create ROS subscribers and publishers.
        # self.get_logger().info(f"Making pubs/subs")
        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.subscription_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.subscription_speed = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)

        self.ttc_threshold = 2 # 2 seconds from hitting wall

    def odom_callback(self, odom_msg):
        # Update current speed
        self.speed = odom_msg.twist.twist.linear.x
    
    def scan_callback(self, scan_msg):
        # Calculating TTC
        # self.get_logger().info(f"Checking TTC")
        if self.speed <= 0.0:
            return

        ranges = scan_msg.ranges
        angles = list(np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment))
        for angle, distance in zip(angles, ranges):
            range_rate = self.speed * np.cos(angle)   # range-rate aka is the range/distance changing (positive = further away, negative = closer). 
            if range_rate <= 0: # if we are going further away no need to brake
                continue
            ttc = distance / range_rate
            if ttc < 2:
                self.get_logger().warn(f"Emergency Brake Engaged ")

                ack_msg = AckermannDriveStamped()
                ack_msg.header.stamp = self.get_clock().now().to_msg()
                ack_msg.drive.speed = 0.0
                ack_msg.drive.steering_angle = 0.0
                self.publisher.publish(ack_msg)
                break
        pass

def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
