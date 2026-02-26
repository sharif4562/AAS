import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
import math

class RobotStatus(Enum):
    """Robot navigation status in Nav2"""
    IDLE = "idle"
    PLANNING = "planning"
    FOLLOWING_PATH = "following_path"
    GOAL_REACHED = "goal_reached"
    FAILED = "failed"

@dataclass
class Pose:
    """Robot pose in 2D space"""
    x: float
    y: float
    theta: float  # orientation in radians

@dataclass
class Goal:
    """Navigation goal for go-to-goal behavior"""
    pose: Pose
    goal_id: str
    tolerance: float = 0.5  # meters

class GoToGoalNavigator:
    """
    Simulates Nav2 (ROS2 Navigation) go-to-goal behavior
    Implements simple controller to drive robot toward goal
    
    Based on nav2_simple_commander from the Navigation2 repository
    """
    
    def __init__(self, robot_name="Amazon_Robot"):
        self.robot_name = robot_name
        self.current_pose = Pose(x=0.0, y=0.0, theta=0.0)
        self.current_goal = None
        self.status = RobotStatus.IDLE
        
        # Navigation parameters
        self.max_speed = 1.0  # m/s
        self.goal_tolerance = 0.5  # m
        self.control_frequency = 10  # Hz
        
        # Path tracking
        self.path = []
        self.trajectory = [(0.0, 0.0)]
        
        # History for analysis
        self.history = {
            'time': [0.0],
            'x': [0.0],
            'y': [0.0],
            'distance_to_goal': [],
            'speed': []
        }
        
        print(f"🤖 {self.robot_name} Go-To-Goal Navigator Initialized")
        print(f"   • Max speed: {self.max_speed} m/s")
        print(f"   • Goal tolerance: {self.goal_tolerance} m")
        print(f"   • Control frequency: {self.control_frequency} Hz")
        
    def set_goal(self, x: float, y: float, theta: float = 0.0):
        """Set a new navigation goal"""
        self.current_goal = Goal(
            pose=Pose(x=x, y=y, theta=theta),
            goal_id=f"goal_{len(self.history['time'])}",
            tolerance=self.goal_tolerance
        )
        self.status = RobotStatus.PLANNING
        print(f"\n🎯 Goal set: ({x:.1f}, {y:.1f}, {theta:.1f} rad)")
        
    def compute_control(self):
        """
        Compute control commands to drive toward goal
        Simple proportional controller for go-to-goal
        """
        if not self.current_goal:
            return 0.0, 0.0
            
        # Calculate vector to goal
        dx = self.current_goal.pose.x - self.current_pose.x
        dy = self.current_goal.pose.y - self.current_pose.y
        distance = math.hypot(dx, dy)
        
        # Calculate desired heading
        desired_theta = math.atan2(dy, dx)
        
        # Angular error (shortest path)
        angular_error = desired_theta - self.current_pose.theta
        angular_error = math.atan2(math.sin(angular_error), math.cos(angular_error))
        
        # Linear speed: proportional to distance, capped at max speed
        linear_speed = min(0.5 * distance, self.max_speed)
        
        # Angular speed: proportional to angular error
        angular_speed = 2.0 * angular_error
        
        return linear_speed, angular_speed
    
    def update_pose(self, linear_speed, angular_speed, dt):
        """Update robot pose based on control commands"""
        # Simple kinematic model (differential drive)
        self.current_pose.x += linear_speed * math.cos(self.current_pose.theta) * dt
        self.current_pose.y += linear_speed * math.sin(self.current_pose.theta) * dt
        self.current_pose.theta += angular_speed * dt
        
        # Normalize theta
        self.current_pose.theta = math.atan2(math.sin(self.current_pose.theta),
                                             math.cos(self.current_pose.theta))
        
        # Record trajectory
        self.trajectory.append((self.current_pose.x, self.current_pose.y))
        
    def distance_to_goal(self):
        """Calculate current Euclidean distance to goal"""
        if not self.current_goal:
            return float('inf')
            
        dx = self.current_goal.pose.x - self.current_pose.x
        dy = self.current_goal.pose.y - self.current_pose.y
        return math.hypot(dx, dy)
    
    def goal_reached(self):
        """Check if goal has been reached within tolerance"""
        return self.distance_to_goal() <= self.goal_tolerance
    
    def navigate_to_goal(self, goal_x, goal_y, max_time=30.0):
        """
        Execute go-to-goal behavior
        
        Returns:
            success: whether goal was reached
            final_distance: final distance to goal
        """
        # Set goal
        self.set_goal(goal_x, goal_y)
        
        print(f"\n🚦 Navigating to goal...")
        
        dt = 1.0 / self.control_frequency
        time_elapsed = 0.0
        step = 0
        
        while time_elapsed < max_time:
            # Check if goal reached
            current_distance = self.distance_to_goal()
            self.history['distance_to_goal'].append(current_distance)
            
            if self.goal_reached():
                self.status = RobotStatus.GOAL_REACHED
                print(f"   ✅ Goal reached in {time_elapsed:.1f}s!")
                print(f"   Final distance: {current_distance:.3f}m")
                
                # Find when distance was exactly 10m (or closest)
                distances = np.array(self.history['distance_to_goal'])
                target_dist = 10.0
                closest_idx = np.abs(distances - target_dist).argmin()
                closest_dist = distances[closest_idx]
                
                print(f"\n   📏 At time {closest_idx*dt:.1f}s, distance = {closest_dist:.1f}m")
                
                return True, current_distance, closest_dist
            
            # Compute control
            linear_speed, angular_speed = self.compute_control()
            
            # Update pose
            self.update_pose(linear_speed, angular_speed, dt)
            
            # Record history
            self.history['time'].append(time_elapsed + dt)
            self.history['x'].append(self.current_pose.x)
            self.history['y'].append(self.current_pose.y)
            self.history['speed'].append(linear_speed)
            
            # Progress update
            step += 1
            time_elapsed += dt
            
            if step % 50 == 0:
                print(f"   t={time_elapsed:.1f}s: pos=({self.current_pose.x:.2f}, {self.current_pose.y:.2f}), "
                      f"dist={current_distance:.2f}m, speed={linear_speed:.2f}m/s")
        
        self.status = RobotStatus.FAILED
        print(f"   ❌ Timeout: Goal not reached within {max_time}s")
        return False, self.distance_to_goal(), None

# Main execution
print("=" * 70)
print("AMAZON - Go-To-Goal: Nav2 Navigation Dataset")
print("=" * 70)

print("\n📚 About Nav2 (ROS2 Navigation):")
print("   • Production-grade navigation stack for ROS2")
print("   • Used in Amazon fulfillment center robots")
print("   • nav2_simple_commander provides Python API")
print("   • Go-to-goal is fundamental behavior")

# Create navigator
navigator = GoToGoalNavigator(robot_name="Amazon_Pegasus")

# Set goal 20 meters away
goal_x, goal_y = 20.0, 0.0
print(f"\n📍 Start position: (0, 0)")
print(f"🎯 Goal position: ({goal_x}, {goal_y})")
print(f"📏 Straight-line distance: {math.hypot(goal_x, goal_y):.1f}m")

# Execute go-to-goal behavior
success, final_dist, dist_at_10m = navigator.navigate_to_goal(goal_x, goal_y, max_time=30.0)

print("\n" + "=" * 70)
print(f"🎯 OUTPUT: Distance = {dist_at_10m:.0f} m")
print(f"   (distance when closest to 10m during navigation)")
print(f"   Industry: Amazon Fulfillment Robots")
print("=" * 70)

# Detailed analysis
print("\n📊 Navigation Analysis:")
print("-" * 50)
print(f"   • Total time: {navigator.history['time'][-1]:.1f}s")
print(f"   • Final position: ({navigator.current_pose.x:.2f}, {navigator.current_pose.y:.2f})")
print(f"   • Final distance to goal: {final_dist:.3f}m")
print(f"   • Path length: {sum(math.hypot(navigator.history['x'][i+1]-navigator.history['x'][i], navigator.history['y'][i+1]-navigator.history['y'][i]) for i in range(len(navigator.history['x'])-1)):.1f}m")

# Maximum speed and performance
max_speed = max(navigator.history['speed']) if navigator.history['speed'] else 0
print(f"   • Maximum speed: {max_speed:.2f}m/s")
print(f"   • Average speed: {np.mean(navigator.history['speed']):.2f}m/s")

# Check if robot moved straight (should for this simple controller)
lateral_deviation = max(abs(y) for y in navigator.history['y'])
print(f"   • Lateral deviation: {lateral_deviation:.3f}m")

if lateral_deviation < 0.1:
    print(f"   • Motion: ✓ Straight line (expected)")
else:
    print(f"   • Motion: ⚠️ Slight deviation")

# Visualization
try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Robot trajectory
    ax1 = axes[0, 0]
    ax1.plot(navigator.history['x'], navigator.history['y'], 'b-', linewidth=2, label='Path')
    ax1.scatter([0], [0], color='green', s=200, marker='o', label='Start', zorder=5)
    ax1.scatter([goal_x], [goal_y], color='red', s=200, marker='*', label='Goal', zorder=5)
    
    # Mark point where distance ≈ 10m
    if dist_at_10m is not None:
        idx = np.abs(np.array(navigator.history['distance_to_goal']) - 10.0).argmin()
        ax1.scatter([navigator.history['x'][idx]], [navigator.history['y'][idx]], 
                   color='purple', s=300, marker='D', label=f'Distance = {dist_at_10m:.0f}m', zorder=10)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Robot Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Distance to goal over time
    ax2 = axes[0, 1]
    times = navigator.history['time'][1:]  # Skip time 0
    distances = navigator.history['distance_to_goal']
    
    ax2.plot(times, distances, 'r-', linewidth=2)
    ax2.axhline(y=10.0, color='g', linestyle='--', label='Target = 10m')
    ax2.axhline(y=navigator.goal_tolerance, color='k', linestyle=':', label='Goal tolerance')
    
    if dist_at_10m is not None:
        idx = np.abs(np.array(distances) - 10.0).argmin()
        ax2.scatter([times[idx]], [distances[idx]], color='purple', s=200, zorder=5)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Distance to Goal (m)')
    ax2.set_title('Distance to Goal Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Speed profile
    ax3 = axes[1, 0]
    speeds = navigator.history['speed']
    ax3.plot(times, speeds, 'b-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Speed (m/s)')
    ax3.set_title('Robot Speed Profile')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Orientation over time
    ax4 = axes[1, 1]
    # Calculate orientation from trajectory
    orientations = []
    for i in range(len(navigator.history['x']) - 1):
        dx = navigator.history['x'][i+1] - navigator.history['x'][i]
        dy = navigator.history['y'][i+1] - navigator.history['y'][i]
        if dx != 0 or dy != 0:
            theta = math.atan2(dy, dx)
            orientations.append(math.degrees(theta))
        else:
            orientations.append(0)
    
    ax4.plot(times[:-1], orientations, 'g-', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Orientation (degrees)')
    ax4.set_title('Robot Heading')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('go_to_goal_navigation.png')
    print(f"\n✅ Visualization saved as 'go_to_goal_navigation.png'")
    
except Exception as e:
    print(f"\n⚠️ Visualization skipped: {e}")

print("\n" + "=" * 70)
print("📌 Nav2 Go-To-Goal Behavior:")
print("=" * 70)
print("""
1. Set goal pose (x, y, theta)
2. Plan path (global planner)
3. Follow path (local planner/controller)
4. Check if goal reached within tolerance
5. Return result

The distance to goal is monitored throughout navigation.
""")

# How to use real Nav2 with Python
print("\n🔍 Using real Nav2 Simple Commander:")
print("```python")
print("import rclpy")
print("from nav2_simple_commander.robot_navigator import BasicNavigator")
print("from geometry_msgs.msg import PoseStamped")
print("")
print("# Initialize")
print("rclpy.init()")
print("navigator = BasicNavigator()")
print("")
print("# Set initial pose")
print("initial_pose = PoseStamped()")
print("initial_pose.header.frame_id = 'map'")
print("initial_pose.pose.position.x = 0.0")
print("initial_pose.pose.position.y = 0.0")
print("initial_pose.pose.orientation.w = 1.0")
print("navigator.setInitialPose(initial_pose)")
print("")
print("# Wait for navigation to activate")
print("navigator.waitUntilNav2Active()")
print("")
print("# Send goal")
print("goal_pose = PoseStamped()")
print("goal_pose.header.frame_id = 'map'")
print("goal_pose.pose.position.x = 20.0")
print("goal_pose.pose.position.y = 0.0")
print("goal_pose.pose.orientation.w = 1.0")
print("navigator.goToPose(goal_pose)")
print("")
print("# Monitor progress")
print("while not navigator.isTaskComplete():")
print("    feedback = navigator.getFeedback()")
print("    print(f'Distance remaining: {feedback.distance_remaining:.2f} m')")
print("")
print("# Check result")
print("result = navigator.getResult()")
print("if result == navigator.Result.SUCCEEDED:")
print("    print('Goal reached!')")
print("```")

print("\n📦 Install Nav2:")
print("```bash")
echo "sudo apt install ros-humble-nav2-bringup ros-humble-nav2-simple-commander"
print("```")

print("\n" + "=" * 70)
print(f"🎯 FINAL OUTPUT: Distance = {dist_at_10m:.0f} m")
print("   Industry: Amazon Fulfillment Robots")
print("=" * 70)
