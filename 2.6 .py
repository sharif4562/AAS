#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class TurtleBot3Publisher(Node):
    """Publisher node that sends movement commands to TurtleBot3"""
    
    def __init__(self):
        super().__init__('turtlebot3_commander')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('TurtleBot3 Commander Node Initialized')
        
    def send_move_command(self, linear_speed=0.2, angular_speed=0.0, duration=5.0):
        """Publish Twist message to move the robot"""
        
        # Create Twist message
        move_cmd = Twist()
        move_cmd.linear.x = linear_speed    # Move forward
        move_cmd.angular.z = angular_speed   # No rotation
        
        # Publish command
        self.get_logger().info(f'Publishing: linear.x={linear_speed}, angular.z={angular_speed}')
        self.publisher_.publish(move_cmd)
        
        # Keep publishing for specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            self.publisher_.publish(move_cmd)
            time.sleep(0.1)  # 10Hz publish rate
            
        # Stop the robot
        stop_cmd = Twist()  # All zeros by default
        self.publisher_.publish(stop_cmd)
        self.get_logger().info('Robot stopped')

class TurtleBot3Subscriber(Node):
    """Subscriber node that monitors robot movement"""
    
    def __init__(self):
        super().__init__('turtlebot3_monitor')
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_callback,
            10)
        self.get_logger().info('TurtleBot3 Monitor Node Initialized')
        
    def cmd_callback(self, msg):
        """Callback when Twist message is received"""
        self.get_logger().info(f'Executing command: linear.x={msg.linear.x}, angular.z={msg.angular.z}')

def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create publisher and subscriber nodes
    publisher_node = TurtleBot3Publisher()
    subscriber_node = TurtleBot3Subscriber()
    
    # Create multi-threaded executor to run both nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(publisher_node)
    executor.add_node(subscriber_node)
    
    try:
        # Spin the executor in a separate thread
        executor.spin()
    except KeyboardInterrupt:
        publisher_node.get_logger().info('Keyboard interrupt received')
    finally:
        # Clean shutdown
        executor.shutdown()
        publisher_node.destroy_node()
        subscriber_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    # Simple demo without spinning (for single command)
    rclpy.init()
    
    # Create publisher only
    node = TurtleBot3Publisher()
    
    # Send movement command
    node.get_logger().info(' Google Robotics - ROS2 Communication Demo')
    node.get_logger().info('Publishing /cmd_vel to TurtleBot3...')
    
    # Procedure: Publish Twist → Subscribe and Execute → Robot Moves
    node.send_move_command(linear_speed=0.22, angular_speed=0.0, duration=3.0)
    
    node.get_logger().info(' Output: Robot Moves')
    node.get_logger().info(' Industry: Google Robotics')
    
    # Cleanup
    node.destroy_node()
    rclpy.shutdown()
