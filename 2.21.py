import numpy as np
import time
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional

class RobotStatus(Enum):
    """Robot navigation status"""
    IDLE = "idle"
    PLANNING = "planning"
    FOLLOWING_PATH = "following_path"
    AVOIDING_OBSTACLE = "avoiding_obstacle"
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
    """Navigation goal"""
    pose: Pose
    tolerance: float = 0.5  # meters
    goal_id: str = None

class TeslaNavigator:
    """
    Simulates Nav2 (ROS2 Navigation) behavior for Tesla autonomous driving
    Includes path planning, following, and obstacle avoidance
    """
    
    def __init__(self, robot_name="Tesla_Model3"):
        self.robot_name = robot_name
        self.current_pose = Pose(x=0.0, y=0.0, theta=0.0)
        self.current_goal = None
        self.status = RobotStatus.IDLE
        self.path = []
        self.obstacles = []
        self.logs = []
        
        # Navigation parameters
        self.speed = 2.0  # m/s
        self.angular_speed = 0.5  # rad/s
        self.goal_tolerance = 0.5
        self.planning_frequency = 10  # Hz
        self.control_frequency = 50  # Hz
        
        print(f"🚗 {self.robot_name} Navigation System Initialized")
        print(f"   • Speed: {self.speed} m/s")
        print(f"   • Goal tolerance: {self.goal_tolerance} m")
        
    def set_goal(self, x: float, y: float, theta: float = 0.0):
        """Set a new navigation goal"""
        self.current_goal = Goal(
            pose=Pose(x=x, y=y, theta=theta),
            tolerance=self.goal_tolerance,
            goal_id=f"goal_{int(time.time())}"
        )
        self.status = RobotStatus.PLANNING
        self.log_event(f"New goal set: ({x:.1f}, {y:.1f})")
        print(f"\n🎯 New goal: ({x:.1f}, {y:.1f}, {theta:.1f} rad)")
        
    def plan_path(self):
        """Plan path from current pose to goal (simulates Nav2 planner)"""
        if not self.current_goal:
            self.log_event("No goal set for planning")
            return False
            
        self.log_event("Planning path to goal...")
        print("🗺️  Path planning...")
        
        # Simulate simple path planning (straight line with obstacle avoidance)
        start = (self.current_pose.x, self.current_pose.y)
        goal = (self.current_goal.pose.x, self.current_goal.pose.y)
        
        # Check if obstacles block direct path
        direct_path_blocked = self.check_obstacles_on_path(start, goal)
        
        if direct_path_blocked:
            # Generate waypoints to avoid obstacles
            self.path = self.generate_avoidance_path(start, goal)
            print(f"   ⚠️ Obstacles detected, generating alternative path")
        else:
            # Direct path is clear
            self.path = [start, goal]
            print(f"   ✅ Direct path clear")
        
        self.status = RobotStatus.FOLLOWING_PATH
        self.log_event(f"Path planned with {len(self.path)} waypoints")
        return True
    
    def check_obstacles_on_path(self, start: Tuple, goal: Tuple) -> bool:
        """Check if obstacles block the direct path"""
        for obs in self.obstacles:
            # Simple line-obstacle intersection check
            if self.line_intersects_circle(start, goal, obs):
                return True
        return False
    
    def line_intersects_circle(self, start: Tuple, end: Tuple, 
                                obstacle: Tuple[float, float, float]) -> bool:
        """Check if line segment intersects circular obstacle"""
        x1, y1 = start
        x2, y2 = end
        ox, oy, radius = obstacle
        
        # Vector from start to end
        dx = x2 - x1
        dy = y2 - y1
        fx = x1 - ox
        fy = y1 - oy
        
        a = dx*dx + dy*dy
        if a == 0:
            return math.hypot(x1 - ox, y1 - oy) <= radius
            
        b = 2 * (fx*dx + fy*dy)
        c = fx*fx + fy*fy - radius*radius
        discriminant = b*b - 4*a*c
        
        if discriminant >= 0:
            discriminant = math.sqrt(discriminant)
            t1 = (-b - discriminant) / (2*a)
            t2 = (-b + discriminant) / (2*a)
            
            if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                return True
        return False
    
    def generate_avoidance_path(self, start: Tuple, goal: Tuple) -> List[Tuple]:
        """Generate waypoints to avoid obstacles"""
        path = [start]
        
        # Simple obstacle avoidance: add intermediate points
        # In reality, this would use A*, Dijkstra, or RRT
        mid_x = (start[0] + goal[0]) / 2
        mid_y = (start[1] + goal[1]) / 2
        
        # Add waypoint offset to avoid obstacles
        offset = 3.0
        waypoints = [
            (start[0] + (goal[0]-start[0])*0.3, start[1] + (goal[1]-start[1])*0.3 - offset),
            (start[0] + (goal[0]-start[0])*0.7, start[1] + (goal[1]-start[1])*0.7 + offset),
        ]
        
        path.extend(waypoints)
        path.append(goal)
        
        return path
    
    def follow_path(self):
        """Follow the planned path (simulates Nav2 controller)"""
        if not self.path or len(self.path) < 2:
            self.status = RobotStatus.FAILED
            self.log_event("Invalid path for following")
            return False
        
        print(f"\n🚦 Following path with {len(self.path)-1} segments...")
        
        for i in range(len(self.path) - 1):
            segment_start = self.path[i]
            segment_end = self.path[i+1]
            
            print(f"   Segment {i+1}: {segment_start} → {segment_end}")
            
            # Simulate moving along segment
            distance = math.hypot(segment_end[0] - segment_start[0],
                                 segment_end[1] - segment_start[1])
            segment_time = distance / self.speed
            
            # Update current pose (simplified, assumes perfect following)
            self.current_pose.x = segment_end[0]
            self.current_pose.y = segment_end[1]
            
            # Check for new obstacles during navigation
            if self.check_for_dynamic_obstacles():
                print(f"   ⚠️ Dynamic obstacle detected! Replanning...")
                self.status = RobotStatus.AVOIDING_OBSTACLE
                return False
            
            time.sleep(0.1)  # Simulate computation time
        
        # Check if reached goal
        if self.goal_reached():
            self.status = RobotStatus.GOAL_REACHED
            self.log_event("Goal reached successfully!")
            return True
        else:
            self.status = RobotStatus.FAILED
            return False
    
    def check_for_dynamic_obstacles(self) -> bool:
        """Simulate detection of dynamic obstacles"""
        # 10% chance of detecting a new obstacle during navigation
        return np.random.random() < 0.1
    
    def goal_reached(self) -> bool:
        """Check if current pose is within tolerance of goal"""
        if not self.current_goal:
            return False
            
        dx = self.current_pose.x - self.current_goal.pose.x
        dy = self.current_pose.y - self.current_goal.pose.y
        distance = math.hypot(dx, dy)
        
        return distance <= self.current_goal.tolerance
    
    def add_obstacle(self, x: float, y: float, radius: float = 1.0):
        """Add a static obstacle to the environment"""
        self.obstacles.append((x, y, radius))
        self.log_event(f"Obstacle added at ({x}, {y}) with radius {radius}")
        
    def log_event(self, message: str):
        """Log navigation events"""
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        
    def navigate_to_goal(self, x: float, y: float, theta: float = 0.0):
        """Complete navigation pipeline: goal → plan → follow → result"""
        print("\n" + "=" * 70)
        print(f"TESLA - Navigation: Nav2 Simulation")
        print("=" * 70)
        
        # Step 1: Set goal
        self.set_goal(x, y, theta)
        
        # Step 2: Plan path
        if not self.plan_path():
            print("❌ Path planning failed")
            return False
        
        # Step 3: Follow path (with possible replanning)
        max_attempts = 3
        for attempt in range(max_attempts):
            print(f"\n📋 Attempt {attempt + 1}/{max_attempts}")
            
            if self.follow_path():
                break
            else:
                if self.status == RobotStatus.AVOIDING_OBSTACLE:
                    print("   Replanning...")
                    self.plan_path()
                else:
                    print("   Navigation failed")
                    break
        
        # Step 4: Check result
        print("\n" + "-" * 50)
        if self.status == RobotStatus.GOAL_REACHED:
            print("✅ OUTPUT: Goal Reached")
            print(f"   Industry: Tesla Autopilot")
            print(f"\n   Final position: ({self.current_pose.x:.2f}, {self.current_pose.y:.2f})")
            print(f"   Goal position: ({self.current_goal.pose.x:.2f}, {self.current_goal.pose.y:.2f})")
            
            # Calculate accuracy
            dx = self.current_pose.x - self.current_goal.pose.x
            dy = self.current_pose.y - self.current_goal.pose.y
            error = math.hypot(dx, dy)
            print(f"   Position error: {error:.3f} m (tolerance: {self.goal_tolerance} m)")
            
            return True
        else:
            print("❌ OUTPUT: Goal Not Reached")
            print(f"   Industry: Tesla Autopilot")
            print(f"   Status: {self.status.value}")
            return False

# Main execution
if __name__ == "__main__":
    
    # Initialize Tesla navigator
    navigator = TeslaNavigator(robot_name="Tesla_Autopilot")
    
    # Add some static obstacles to the environment
    navigator.add_obstacle(5, 2, 1.5)   # Obstacle 1
    navigator.add_obstacle(8, 5, 2.0)   # Obstacle 2
    navigator.add_obstacle(12, 3, 1.0)  # Obstacle 3
    
    print("\n🌍 Environment:")
    print(f"   • {len(navigator.obstacles)} static obstacles")
    for i, (ox, oy, r) in enumerate(navigator.obstacles):
        print(f"     Obstacle {i+1}: at ({ox}, {oy}), radius {r}m")
    
    # Test Case 1: Simple goal in free space
    print("\n" + "=" * 70)
    print("TEST CASE 1: Simple goal (clear path)")
    navigator.navigate_to_goal(x=10.0, y=5.0, theta=0.0)
    
    # Test Case 2: Goal behind obstacles
    print("\n" + "=" * 70)
    print("TEST CASE 2: Goal behind obstacles (requires avoidance)")
    navigator = TeslaNavigator(robot_name="Tesla_Autopilot")
    navigator.add_obstacle(5, 0, 2.0)
    navigator.add_obstacle(7, 3, 1.5)
    navigator.add_obstacle(9, -2, 1.0)
    
    navigator.navigate_to_goal(x=15.0, y=0.0, theta=0.0)
    
    # Show navigation logs
    print("\n" + "=" * 70)
    print("📋 Navigation Logs:")
    print("-" * 50)
    for log in navigator.logs[-10:]:  # Last 10 logs
        print(log)
    
    # Performance metrics
    print("\n" + "=" * 70)
    print("📊 Navigation Performance Metrics:")
    print("-" * 50)
    
    # Calculate path efficiency if goal reached
    if navigator.status == RobotStatus.GOAL_REACHED and navigator.path:
        straight_line = math.hypot(
            navigator.current_goal.pose.x - 0,
            navigator.current_goal.pose.y - 0
        )
        
        path_length = 0
        for i in range(len(navigator.path) - 1):
            path_length += math.hypot(
                navigator.path[i+1][0] - navigator.path[i][0],
                navigator.path[i+1][1] - navigator.path[i][1]
            )
        
        efficiency = straight_line / path_length * 100
        print(f"   Straight-line distance: {straight_line:.2f} m")
        print(f"   Actual path length: {path_length:.2f} m")
        print(f"   Path efficiency: {efficiency:.1f}%")
        print(f"   Extra distance: {path_length - straight_line:.2f} m")
    
    print("\n" + "=" * 70)
    print("📌 About Nav2 (ROS2 Navigation):")
    print("   • Production-grade navigation stack for ROS2")
    print("   • Used in autonomous robots worldwide")
    print("   • Includes: mapping, localization, planning, control")
    print("   • Pluggable architecture with multiple algorithms")
    print("=" * 70)

# Reference to Nav2 documentation
print("\n📚 Nav2 Resources:")
print("   • Official site: https://navigation.ros.org")
print("   • GitHub: https://github.com/ros-planning/navigation2")
print("   • Tutorials: https://navigation.ros.org/getting_started/index.html")
print("   • ROS 2 Foxy, Galactic, Humble, Rolling releases")

# How to use real Nav2 with ROS2
print("\n🔍 Example using real Nav2 with ROS2:")
print("```bash")
print("# Install Nav2")
print("sudo apt install ros-humble-nav2-bringup")
print("")
print("# Launch Nav2 with TurtleBot3 simulation")
print("ros2 launch nav2_bringup tb3_simulation_launch.py headless:=False")
print("")
print("# Send navigation goal via RViz2 or command line")
print("ros2 topic pub /goal_pose geometry_msgs/PoseStamped \"{")
print("  header: {frame_id: 'map'},")
print("  pose: {position: {x: 2.0, y: 2.0, z: 0.0}, orientation: {w: 1.0}}")
print("}\"")
print("```")

print("\n🐍 Python code to send goals to Nav2:")
print("```python")
print("import rclpy")
print("from geometry_msgs.msg import PoseStamped")
print("from nav2_simple_commander.robot_navigator import BasicNavigator")
print("")
print("def send_goal():")
print("    rclpy.init()")
print("    navigator = BasicNavigator()")
print("    ")
print("    # Set initial pose")
print("    initial_pose = PoseStamped()")
print("    initial_pose.header.frame_id = 'map'")
print("    initial_pose.pose.position.x = 0.0")
print("    initial_pose.pose.position.y = 0.0")
print("    navigator.setInitialPose(initial_pose)")
print("    ")
print("    # Wait for navigation to activate")
print("    navigator.waitUntilNav2Active()")
print("    ")
print("    # Send goal")
print("    goal_pose = PoseStamped()")
print("    goal_pose.header.frame_id = 'map'")
print("    goal_pose.pose.position.x = 5.0")
print("    goal_pose.pose.position.y = 3.0")
print("    navigator.goToPose(goal_pose)")
print("    ")
print("    # Monitor progress")
print("    while not navigator.isTaskComplete():")
print("        feedback = navigator.getFeedback()")
print("        print(f\"Distance remaining: {feedback.distance_remaining:.2f}\")")
print("    ")
print("    print('Goal Reached!')")
print("```")
