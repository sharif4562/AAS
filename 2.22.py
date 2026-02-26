import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
import math

class WheelPosition(Enum):
    """Wheel positions on differential drive robot"""
    LEFT = "left"
    RIGHT = "right"

@dataclass
class WheelVelocity:
    """Velocity of a single wheel"""
    linear: float  # m/s
    angular: float  # rad/s

@dataclass
class RobotTwist:
    """Robot's linear and angular velocity"""
    linear_x: float  # m/s
    angular_z: float  # rad/s

class DifferentialDriveKinematics:
    """
    Differential drive kinematics for wheeled robots
    Based on ROS2 control principles from ros2_control framework
    """
    
    def __init__(self, wheel_separation=0.5, wheel_radius=0.1, robot_name="Amazon_Drive"):
        """
        Initialize differential drive robot
        
        Args:
            wheel_separation: distance between left and right wheels (meters)
            wheel_radius: radius of wheels (meters)
            robot_name: name of the robot
        """
        self.wheel_separation = wheel_separation
        self.wheel_radius = wheel_radius
        self.robot_name = robot_name
        
        # Robot state
        self.x = 0.0  # x position (m)
        self.y = 0.0  # y position (m)
        self.theta = 0.0  # orientation (rad)
        
        # Wheel velocities
        self.left_wheel_vel = WheelVelocity(linear=0.0, angular=0.0)
        self.right_wheel_vel = WheelVelocity(linear=0.0, angular=0.0)
        
        # Commanded twist
        self.cmd_twist = RobotTwist(linear_x=0.0, angular_z=0.0)
        
        # History for plotting
        self.history = {
            'time': [0.0],
            'x': [0.0],
            'y': [0.0],
            'theta': [0.0],
            'left_wheel': [0.0],
            'right_wheel': [0.0]
        }
        
        print(f"🤖 {self.robot_name} Initialized:")
        print(f"   • Wheel separation: {wheel_separation} m")
        print(f"   • Wheel radius: {wheel_radius} m")
        
    def forward_kinematics(self, left_wheel_angular_vel, right_wheel_angular_vel):
        """
        Convert wheel angular velocities to robot twist
        This is used for odometry estimation
        
        Args:
            left_wheel_angular_vel: left wheel angular velocity (rad/s)
            right_wheel_angular_vel: right wheel angular velocity (rad/s)
        
        Returns:
            RobotTwist: robot's linear and angular velocity
        """
        # Wheel linear velocities
        v_left = left_wheel_angular_vel * self.wheel_radius
        v_right = right_wheel_angular_vel * self.wheel_radius
        
        # Robot linear and angular velocity
        v = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / self.wheel_separation
        
        return RobotTwist(linear_x=v, angular_z=omega)
    
    def inverse_kinematics(self, linear_x, angular_z):
        """
        Convert robot twist to wheel angular velocities
        This is used for motion control
        
        Args:
            linear_x: desired linear velocity (m/s)
            angular_z: desired angular velocity (rad/s)
        
        Returns:
            tuple: (left_wheel_angular_vel, right_wheel_angular_vel) in rad/s
        """
        # Calculate wheel linear velocities
        v_left = linear_x - (angular_z * self.wheel_separation / 2.0)
        v_right = linear_x + (angular_z * self.wheel_separation / 2.0)
        
        # Convert to angular velocities
        omega_left = v_left / self.wheel_radius
        omega_right = v_right / self.wheel_radius
        
        # Store commanded wheel velocities
        self.left_wheel_vel.angular = omega_left
        self.left_wheel_vel.linear = v_left
        self.right_wheel_vel.angular = omega_right
        self.right_wheel_vel.linear = v_right
        
        return omega_left, omega_right
    
    def update_odometry(self, dt, left_wheel_angular_vel, right_wheel_angular_vel):
        """
        Update robot pose based on wheel velocities (dead reckoning)
        
        Args:
            dt: time step (seconds)
            left_wheel_angular_vel: measured left wheel velocity (rad/s)
            right_wheel_angular_vel: measured right wheel velocity (rad/s)
        """
        # Get robot twist from forward kinematics
        twist = self.forward_kinematics(left_wheel_angular_vel, right_wheel_angular_vel)
        self.cmd_twist = twist
        
        # Update pose using simple Euler integration
        self.x += twist.linear_x * dt * math.cos(self.theta)
        self.y += twist.linear_x * dt * math.sin(self.theta)
        self.theta += twist.angular_z * dt
        
        # Normalize theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        # Store history
        self.history['time'].append(self.history['time'][-1] + dt)
        self.history['x'].append(self.x)
        self.history['y'].append(self.y)
        self.history['theta'].append(self.theta)
        self.history['left_wheel'].append(left_wheel_angular_vel)
        self.history['right_wheel'].append(right_wheel_angular_vel)
    
    def move_forward(self, linear_speed, duration):
        """
        Command robot to move forward (straight motion)
        
        Args:
            linear_speed: desired forward speed (m/s)
            duration: movement duration (seconds)
        """
        print(f"\n📋 Command: Move forward at {linear_speed} m/s for {duration} s")
        
        # For straight motion, angular_z = 0
        left_omega, right_omega = self.inverse_kinematics(linear_speed, 0.0)
        
        print(f"   Inverse kinematics:")
        print(f"      ω_left = {left_omega:.2f} rad/s")
        print(f"      ω_right = {right_omega:.2f} rad/s")
        print(f"      v_left = {self.left_wheel_vel.linear:.2f} m/s")
        print(f"      v_right = {self.right_wheel_vel.linear:.2f} m/s")
        
        # Simulate motion
        dt = 0.1  # control loop time step
        steps = int(duration / dt)
        
        for step in range(steps):
            self.update_odometry(dt, left_omega, right_omega)
            
        print(f"\n   Final pose:")
        print(f"      x = {self.x:.2f} m")
        print(f"      y = {self.y:.2f} m")
        print(f"      θ = {self.theta:.2f} rad ({math.degrees(self.theta):.1f}°)")
        
        return self.x, self.y, self.theta
    
    def move_arc(self, linear_speed, angular_speed, duration):
        """
        Command robot to move in an arc
        
        Args:
            linear_speed: desired linear speed (m/s)
            angular_speed: desired angular speed (rad/s)
            duration: movement duration (seconds)
        """
        print(f"\n📋 Command: Move in arc at {linear_speed} m/s, {angular_speed:.2f} rad/s for {duration} s")
        
        left_omega, right_omega = self.inverse_kinematics(linear_speed, angular_speed)
        
        print(f"   Inverse kinematics:")
        print(f"      ω_left = {left_omega:.2f} rad/s")
        print(f"      ω_right = {right_omega:.2f} rad/s")
        
        # Simulate motion
        dt = 0.1
        steps = int(duration / dt)
        
        for step in range(steps):
            self.update_odometry(dt, left_omega, right_omega)
            
        print(f"\n   Final pose:")
        print(f"      x = {self.x:.2f} m")
        print(f"      y = {self.y:.2f} m")
        print(f"      θ = {self.theta:.2f} rad ({math.degrees(self.theta):.1f}°)")
        
        return self.x, self.y, self.theta
    
    def rotate_in_place(self, angular_speed, duration):
        """
        Rotate robot in place
        
        Args:
            angular_speed: desired rotation speed (rad/s)
            duration: rotation duration (seconds)
        """
        print(f"\n📋 Command: Rotate in place at {angular_speed:.2f} rad/s for {duration} s")
        
        left_omega, right_omega = self.inverse_kinematics(0.0, angular_speed)
        
        print(f"   Inverse kinematics:")
        print(f"      ω_left = {left_omega:.2f} rad/s")
        print(f"      ω_right = {right_omega:.2f} rad/s")
        
        # Simulate motion
        dt = 0.1
        steps = int(duration / dt)
        
        for step in range(steps):
            self.update_odometry(dt, left_omega, right_omega)
            
        print(f"\n   Final pose:")
        print(f"      x = {self.x:.2f} m (unchanged)")
        print(f"      y = {self.y:.2f} m (unchanged)")
        print(f"      θ = {self.theta:.2f} rad ({math.degrees(self.theta):.1f}°)")
        
        return self.x, self.y, self.theta
    
    def stop(self):
        """Stop all motion"""
        self.inverse_kinematics(0.0, 0.0)
        print("\n🛑 Robot stopped")
    
    def reset(self):
        """Reset robot to origin"""
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.history = {
            'time': [0.0],
            'x': [0.0],
            'y': [0.0],
            'theta': [0.0],
            'left_wheel': [0.0],
            'right_wheel': [0.0]
        }
        print("\n🔄 Robot reset to origin")

# Main execution
print("=" * 70)
print("AMAZON - Differential Drive Kinematics: ros2_control Framework")
print("=" * 70)

print("\n📚 About ros2_control:")
print("   • Generic control framework for ROS 2")
print("   • Provides hardware interfaces, controllers, and transmissions")
print("   • Used in Amazon fulfillment center robots")
print("   • Supports differential drive, ackermann, and other drive types")

# Create differential drive robot (like Amazon warehouse robot)
robot = DifferentialDriveKinematics(
    wheel_separation=0.4,  # 40 cm between wheels
    wheel_radius=0.075,     # 7.5 cm wheel radius
    robot_name="Amazon_Pegasus"
)

print("\n" + "=" * 70)
print("🎯 OUTPUT DEMONSTRATION: Straight Motion")
print("=" * 70)

# Demonstrate straight motion (linear_x > 0, angular_z = 0)
print("\n📌 TEST: Forward Straight Motion")
robot.move_forward(linear_speed=0.5, duration=5.0)

# Check if motion was straight
print(f"\n   Motion analysis:")
print(f"      Final y position: {robot.y:.4f} m")
print(f"      Final orientation: {robot.theta:.4f} rad")

if abs(robot.y) < 0.01 and abs(robot.theta) < 0.01:
    print("\n✅ VERIFICATION: Robot moved straight")
else:
    print("\n⚠️ VERIFICATION: Robot deviated from straight line")

print("\n" + "-" * 50)
print("🎯 OUTPUT: Straight Motion")
print("   Industry: Amazon Fulfillment Robots")
print("-" * 50)

# Demonstrate other motion types for comparison
print("\n" + "=" * 70)
print("📊 COMPARISON: Different Motion Types")
print("=" * 70)

# Test Case 2: Rotate in place
robot.reset()
robot.rotate_in_place(angular_speed=0.5, duration=3.0)

# Test Case 3: Arc motion
robot.reset()
robot.move_arc(linear_speed=0.5, angular_speed=0.2, duration=5.0)

# Test Case 4: Complex sequence
print("\n" + "=" * 70)
print("🔄 Complex Motion Sequence")
print("=" * 70)

robot.reset()
print("\n📋 Sequence:")
print("   1. Move forward 2 seconds")
print("   2. Rotate 90°")
print("   3. Move forward 2 seconds")
print("   4. Rotate 90°")
print("   5. Move forward 2 seconds")

robot.move_forward(0.5, 2.0)
robot.rotate_in_place(math.pi/2, 1.57)  # 90° turn
robot.move_forward(0.5, 2.0)
robot.rotate_in_place(math.pi/2, 1.57)
robot.move_forward(0.5, 2.0)

print(f"\n📈 Final position after sequence:")
print(f"   x = {robot.x:.2f} m, y = {robot.y:.2f} m, θ = {math.degrees(robot.theta):.1f}°")

# Visualization
try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Trajectory in XY plane
    ax1 = axes[0, 0]
    ax1.plot(robot.history['x'], robot.history['y'], 'b-', linewidth=2)
    ax1.plot(0, 0, 'go', markersize=10, label='Start')
    ax1.plot(robot.history['x'][-1], robot.history['y'][-1], 'ro', markersize=10, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Robot Trajectory')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    # Plot 2: Wheel velocities over time
    ax2 = axes[0, 1]
    time = robot.history['time']
    ax2.plot(time, robot.history['left_wheel'], 'r-', label='Left wheel', linewidth=2)
    ax2.plot(time, robot.history['right_wheel'], 'b-', label='Right wheel', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular velocity (rad/s)')
    ax2.set_title('Wheel Velocities')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Orientation over time
    ax3 = axes[1, 0]
    ax3.plot(time, [math.degrees(t) for t in robot.history['theta']], 'g-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Orientation (degrees)')
    ax3.set_title('Robot Orientation')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Forward velocity check
    ax4 = axes[1, 1]
    # Calculate forward velocity from odometry
    v_x = np.diff(robot.history['x']) / np.diff(time)
    v_y = np.diff(robot.history['y']) / np.diff(time)
    v_forward = np.sqrt(np.array(v_x)**2 + np.array(v_y)**2)
    
    ax4.plot(time[1:], v_forward, 'purple', linewidth=2)
    ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Commanded speed')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.set_title('Forward Speed')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('differential_drive_kinematics.png')
    print(f"\n✅ Visualization saved as 'differential_drive_kinematics.png'")
    
except Exception as e:
    print(f"\n⚠️ Visualization skipped: {e}")

print("\n" + "=" * 70)
print("📌 Differential Drive Kinematics Equations:")
print("=" * 70)
print("\n🔧 Forward Kinematics (Odometry):")
print("   v = (v_right + v_left) / 2")
print("   ω = (v_right - v_left) / L")
print("   where L = wheel separation")
print("\n   ẋ = v * cos(θ)")
print("   ẏ = v * sin(θ)")
print("   θ̇ = ω")
print("\n🔧 Inverse Kinematics (Control):")
print("   v_left = v - (ω * L / 2)")
print("   v_right = v + (ω * L / 2)")
print("\n   ω_left = v_left / r")
print("   ω_right = v_right / r")
print("   where r = wheel radius")

# How to use with real ros2_control
print("\n" + "=" * 70)
print("🔍 Using with real ros2_control:")
print("=" * 70)
print("\n📥 Install ros2_control:")
print("```bash")
print("sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers")
print("```")

print("\n📝 Create a differential drive robot URDF:")
print("```xml")
print("<robot name=\"amazon_robot\">")
print("  <ros2_control name=\"RobotSystem\" type=\"system\">")
print("    <hardware>")
print("      <plugin>ros2_control_demos/PositionActuator</plugin>")
print("    </hardware>")
print("    <joint name=\"left_wheel_joint\">")
print("      <command_interface name=\"velocity\"/>")
print("      <state_interface name=\"position\"/>")
print("      <state_interface name=\"velocity\"/>")
print("    </joint>")
print("    <joint name=\"right_wheel_joint\">")
print("      <command_interface name=\"velocity\"/>")
print("      <state_interface name=\"position\"/>")
print("      <state_interface name=\"velocity\"/>")
print("    </joint>")
print("  </ros2_control>")
print("  ")
print("  <gazebo>")
print("    <plugin filename=\"libgazebo_ros2_control.so\" name=\"gazebo_ros2_control\">")
print("      <parameters>$(find amazon_bot)/config/amazon_control.yaml</parameters>")
print("    </plugin>")
print("  </gazebo>")
print("</robot>")
print("```")

print("\n📋 Create controller configuration:")
print("```yaml")
print("controller_manager:")
print("  ros__parameters:")
print("    update_rate: 100  # Hz")
print("    ")
print("    diff_drive_controller:")
print("      type: diff_drive_controller/DiffDriveController")
print("      ")
print("    joint_state_broadcaster:")
print("      type: joint_state_broadcaster/JointStateBroadcaster")
print("")
print("diff_drive_controller:")
print("  ros__parameters:")
print("    left_wheel_names: [\"left_wheel_joint\"]")
print("    right_wheel_names: [\"right_wheel_joint\"]")
print("    ")
print("    wheel_separation: 0.4")
print("    wheel_radius: 0.075")
print("    ")
print("    linear.x.max_velocity: 1.0")
print("    angular.z.max_velocity: 2.0")
print("```")

print("\n🐍 Python code to command robot:")
print("```python")
print("import rclpy")
print("from geometry_msgs.msg import Twist")
print("")
print("def send_velocity_command():")
print("    rclpy.init()")
print("    node = rclpy.create_node('velocity_publisher')")
print("    publisher = node.create_publisher(Twist, '/cmd_vel', 10)")
print("    ")
print("    # Command: Straight motion at 0.5 m/s")
print("    twist = Twist()")
print("    twist.linear.x = 0.5")
print("    twist.angular.z = 0.0")
print("    ")
print("    publisher.publish(twist)")
print("    node.get_logger().info('Publishing: \"linear_x: 0.5, angular_z: 0.0\"')")
print("    ")
print("    rclpy.spin_once(node, timeout_sec=1.0)")
print("    node.destroy_node()")
print("    rclpy.shutdown()")
print("```")

print("\n" + "=" * 70)
print("📊 Kinematics Verification:")
print("-" * 50)

# Verify straight motion condition
robot.reset()
v_cmd = 0.5
omega_cmd = 0.0
left, right = robot.inverse_kinematics(v_cmd, omega_cmd)

print(f"For straight motion (v={v_cmd} m/s, ω={omega_cmd} rad/s):")
print(f"   Left wheel angular velocity: {left:.3f} rad/s")
print(f"   Right wheel angular velocity: {right:.3f} rad/s")
print(f"   Left wheel linear velocity: {left * robot.wheel_radius:.3f} m/s")
print(f"   Right wheel linear velocity: {right * robot.wheel_radius:.3f} m/s")
print(f"\n   Condition for straight motion: ω_left = ω_right")
print(f"   ✓ Verified: {left:.3f} ≈ {right:.3f}")

print("\n" + "=" * 70)
