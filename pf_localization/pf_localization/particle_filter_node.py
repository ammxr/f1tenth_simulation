import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
import numpy as np
from .particle_filter import ParticleFilter

class PFNode(Node):
    def __init__(self):
        super().__init__('particle_filter_node')
        
        # 1. Define QoS Profiles
        # Map servers usually use 'Transient Local' durability
        map_qos = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            depth=1)

        # 2. Parameters
        self.declare_parameter('num_particles', 200)
        self.num_particles = self.get_parameter('num_particles').value
        
        # 3. Subscriptions (Using the topics from your 'ros2 topic list')
        self.map_sub = self.create_subscription(
            OccupancyGrid, 
            '/map', 
            self.map_callback, 
            map_qos) # Use the specific QoS here
            
        self.scan_sub = self.create_subscription(
            LaserScan, 
            '/scan', 
            self.scan_callback, 
            10)
            
        self.odom_sub = self.create_subscription(
            Odometry, 
            '/ego_racecar/odom', 
            self.odom_callback, 
            10)
        
        # 4. Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/pf/pose', 10)
        self.particles_pub = self.create_publisher(PoseArray, '/pf/particles', 10)
        
        self.pf = None
        self.last_time = self.get_clock().now()
        self.get_logger().info("PF Node Initialized. QoS configured for /map.")

    def map_callback(self, msg):
        if self.pf is None:
            self.get_logger().info(f"SUCCESS: Map Received ({msg.info.width}x{msg.info.height}). Initializing particles...")
            # VAUL-style initialization
            self.pf = ParticleFilter(
                num_particles=self.num_particles,
                x_init=0.0, y_init=0.0, theta_init=0.0,
                map_data=msg
            )

    def odom_callback(self, msg):
        if self.pf is None: return
        
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        
        if dt > 0:
            odom_data = {
                'linear_vel': msg.twist.twist.linear.x,
                'angular_vel': msg.twist.twist.angular.z
            }
            self.pf.predict(odom_data, dt)

    def scan_callback(self, msg):
        if self.pf is None: return
    
        # Pass LiDAR metadata to the filter
        self.pf.update(msg.ranges, msg.angle_min, msg.angle_increment)
        self.pf.resample()
        
        self.publish_visuals(msg.header)
    
    def publish_visuals(self, header):
        # Current Estimate
        est = self.pf.get_estimate()
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = float(est[0])
        pose_msg.pose.position.y = float(est[1])
        pose_msg.pose.orientation.z = np.sin(est[2] / 2.0)
        pose_msg.pose.orientation.w = np.cos(est[2] / 2.0)
        self.pose_pub.publish(pose_msg)
        
        # Particle Cloud
        arr_msg = PoseArray()
        arr_msg.header = header
        arr_msg.header.frame_id = 'map'
        for p in self.pf.particles:
            p_pose = Pose()
            p_pose.position.x = float(p[0])
            p_pose.position.y = float(p[1])
            p_pose.orientation.z = np.sin(p[2] / 2.0)
            p_pose.orientation.w = np.cos(p[2] / 2.0)
            arr_msg.poses.append(p_pose)
        self.particles_pub.publish(arr_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
