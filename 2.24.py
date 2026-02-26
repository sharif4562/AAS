import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.path import Path
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon as ShapelyPolygon, LineString
import random

class ConfigurationSpace:
    """
    Configuration Space (C-Space) for motion planning
    Based on concepts from "Planning Algorithms" by LaValle
    """
    
    def __init__(self, width=20, height=20, robot_radius=1.0):
        """
        Initialize configuration space
        
        Args:
            width: workspace width
            height: workspace height
            robot_radius: robot radius (for Minkowski sum)
        """
        self.width = width
        self.height = height
        self.robot_radius = robot_radius
        
        # Obstacles in workspace
        self.obstacles = []
        
        # Configuration space obstacles (after Minkowski sum)
        self.cspace_obstacles = []
        
        # Start and goal in configuration space
        self.start = None
        self.goal = None
        
        # Path in configuration space
        self.path = []
        
        print(f"🌌 Configuration Space Created:")
        print(f"   • Workspace: {width}m x {height}m")
        print(f"   • Robot radius: {robot_radius}m")
        print(f"   • C-Space dimension: 2 (x, y)")
        
    def add_obstacle(self, obstacle_type, params):
        """
        Add obstacle to workspace
        
        Args:
            obstacle_type: 'rectangle', 'circle', or 'polygon'
            params: obstacle parameters
        """
        self.obstacles.append({
            'type': obstacle_type,
            'params': params
        })
        print(f"   Added {obstacle_type} obstacle: {params}")
        
    def compute_cspace_obstacles(self):
        """
        Compute C-Space obstacles using Minkowski sum
        In C-Space, obstacles are grown by robot radius
        """
        self.cspace_obstacles = []
        
        for obs in self.obstacles:
            if obs['type'] == 'rectangle':
                x, y, w, h = obs['params']
                # Grow rectangle by robot radius
                cspace_obs = {
                    'type': 'rectangle',
                    'params': (x - self.robot_radius, 
                              y - self.robot_radius,
                              w + 2*self.robot_radius,
                              h + 2*self.robot_radius)
                }
                self.cspace_obstacles.append(cspace_obs)
                
            elif obs['type'] == 'circle':
                cx, cy, r = obs['params']
                # Grow circle by robot radius
                cspace_obs = {
                    'type': 'circle',
                    'params': (cx, cy, r + self.robot_radius)
                }
                self.cspace_obstacles.append(cspace_obs)
                
            elif obs['type'] == 'polygon':
                vertices = obs['params']
                # For polygon, we would need proper Minkowski sum
                # Simplified: use bounding box grown by robot radius
                vertices_array = np.array(vertices)
                min_x, min_y = np.min(vertices_array, axis=0)
                max_x, max_y = np.max(vertices_array, axis=0)
                
                cspace_obs = {
                    'type': 'rectangle',
                    'params': (min_x - self.robot_radius,
                              min_y - self.robot_radius,
                              max_x - min_x + 2*self.robot_radius,
                              max_y - min_y + 2*self.robot_radius)
                }
                self.cspace_obstacles.append(cspace_obs)
    
    def set_start_goal(self, start, goal):
        """Set start and goal positions in workspace"""
        self.start = start
        self.goal = goal
        print(f"\n📍 Start: {start}, Goal: {goal}")
        
    def point_in_cspace_obstacle(self, point):
        """Check if a point in C-Space is in collision"""
        x, y = point
        
        for obs in self.cspace_obstacles:
            if obs['type'] == 'rectangle':
                rx, ry, rw, rh = obs['params']
                if rx <= x <= rx + rw and ry <= y <= ry + rh:
                    return True
                    
            elif obs['type'] == 'circle':
                cx, cy, r = obs['params']
                if np.hypot(x - cx, y - cy) <= r:
                    return True
                    
        return False
    
    def line_in_cspace_obstacle(self, p1, p2, num_samples=10):
        """
        Check if line segment between two points collides with C-Space obstacles
        """
        # Sample points along the line
        for t in np.linspace(0, 1, num_samples):
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            
            if self.point_in_cspace_obstacle((x, y)):
                return True
        return False
    
    def generate_path(self, num_waypoints=5):
        """
        Generate a simple path from start to goal
        In practice, this would use RRT, PRM, or A*
        """
        self.path = [self.start]
        
        # Generate intermediate waypoints
        for i in range(1, num_waypoints):
            t = i / num_waypoints
            x = self.start[0] + t * (self.goal[0] - self.start[0])
            y = self.start[1] + t * (self.goal[1] - self.start[1])
            
            # Add some noise to avoid obstacles
            if self.point_in_cspace_obstacle((x, y)):
                # Find alternative point
                for _ in range(10):
                    nx = x + random.uniform(-2, 2)
                    ny = y + random.uniform(-2, 2)
                    
                    if (0 <= nx <= self.width and 0 <= ny <= self.height and
                        not self.point_in_cspace_obstacle((nx, ny))):
                        self.path.append((nx, ny))
                        break
                else:
                    self.path.append((x, y))
            else:
                self.path.append((x, y))
        
        self.path.append(self.goal)
        return self.path
    
    def check_path_collision(self, path=None):
        """
        Check if entire path is collision-free in C-Space
        
        Returns:
            bool: True if path is collision-free
        """
        if path is None:
            path = self.path
            
        if len(path) < 2:
            return False
            
        for i in range(len(path) - 1):
            if self.line_in_cspace_obstacle(path[i], path[i+1]):
                print(f"   ❌ Collision detected between {path[i]} and {path[i+1]}")
                return False
                
        print(f"   ✅ All path segments are collision-free")
        return True
    
    def is_collision_free(self):
        """Check if the current configuration is collision-free"""
        return self.check_path_collision()

# Main execution
print("=" * 70)
print("NVIDIA - Configuration Space: Motion Planning Maps")
print("=" * 70)

print("\n📚 About Configuration Space (C-Space):")
print("   • Robot configuration = point in C-Space")
print("   • Obstacles in workspace → forbidden regions in C-Space")
print("   • Path planning = finding continuous path in free C-Space")
print("   • Minkowski sum: grow obstacles by robot radius")

# Create configuration space
cspace = ConfigurationSpace(width=20, height=20, robot_radius=1.0)

# Add obstacles to workspace
print("\n🧱 Adding Workspace Obstacles:")
cspace.add_obstacle('rectangle', (4, 4, 3, 2))   # Rectangular obstacle
cspace.add_obstacle('circle', (12, 8, 2))        # Circular obstacle
cspace.add_obstacle('rectangle', (8, 12, 4, 1.5)) # Horizontal bar
cspace.add_obstacle('polygon', [(15, 5), (18, 8), (16, 11)]) # Triangle

# Compute C-Space obstacles (Minkowski sum with robot)
print("\n📐 Computing C-Space obstacles (Minkowski sum)...")
cspace.compute_cspace_obstacles()

# Set start and goal
cspace.set_start_goal(start=(1, 1), goal=(18, 18))

# Generate a path
print("\n🛤️  Generating path through C-Space...")
path = cspace.generate_path(num_waypoints=8)
print(f"   Path waypoints: {len(path)} points")

# Check if path is collision-free
print("\n" + "-" * 50)
print("🔍 Checking C-Space for collisions...")

# Check each waypoint
print("\n   Checking waypoints:")
for i, point in enumerate(path):
    in_collision = cspace.point_in_cspace_obstacle(point)
    status = "❌ COLLISION" if in_collision else "✅ Free"
    print(f"      Waypoint {i}: {point} → {status}")

# Check entire path
print("\n   Checking path segments:")
collision_free = cspace.check_path_collision()

print("\n" + "=" * 70)
if collision_free:
    print("🎯 OUTPUT: Collision-Free")
    print(f"   Industry: NVIDIA Autonomous Systems")
    print(f"\n   ✓ The path from {cspace.start} to {cspace.goal}")
    print(f"     is collision-free in configuration space")
else:
    print("❌ OUTPUT: Path in Collision")
    print(f"   Industry: NVIDIA Autonomous Systems")
    print(f"\n   ✗ The path contains collisions with C-Space obstacles")
    print(f"     (path planning failed)")

# Detailed C-Space analysis
print("\n" + "=" * 70)
print("📊 C-Space Analysis")
print("=" * 70)

# Check start and goal configurations
print(f"\n📍 Start configuration: {cspace.start}")
if cspace.point_in_cspace_obstacle(cspace.start):
    print("   ❌ Start in collision! (invalid initial configuration)")
else:
    print("   ✅ Start in free space")

print(f"\n🎯 Goal configuration: {cspace.goal}")
if cspace.point_in_cspace_obstacle(cspace.goal):
    print("   ❌ Goal in collision! (unreachable goal)")
else:
    print("   ✅ Goal in free space")

# Compute free space percentage
total_points = 100
free_points = 0
for _ in range(total_points):
    x = random.uniform(0, cspace.width)
    y = random.uniform(0, cspace.height)
    if not cspace.point_in_cspace_obstacle((x, y)):
        free_points += 1

free_percentage = (free_points / total_points) * 100
print(f"\n📈 Free C-Space: {free_percentage:.1f}% of workspace")
print(f"   (sampled from {total_points} random points)")

# Visualization
try:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Workspace (with original obstacles)
    ax1 = axes[0]
    ax1.set_xlim(0, cspace.width)
    ax1.set_ylim(0, cspace.height)
    ax1.set_aspect('equal')
    ax1.set_title('Workspace (Original Obstacles)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    # Draw original obstacles
    for obs in cspace.obstacles:
        if obs['type'] == 'rectangle':
            x, y, w, h = obs['params']
            rect = Rectangle((x, y), w, h, facecolor='red', alpha=0.5, edgecolor='darkred')
            ax1.add_patch(rect)
        elif obs['type'] == 'circle':
            cx, cy, r = obs['params']
            circle = Circle((cx, cy), r, facecolor='red', alpha=0.5, edgecolor='darkred')
            ax1.add_patch(circle)
        elif obs['type'] == 'polygon':
            vertices = obs['params']
            poly = Polygon(vertices, closed=True, facecolor='red', alpha=0.5, edgecolor='darkred')
            ax1.add_patch(poly)
    
    # Draw robot (as circle at start)
    robot = Circle(cspace.start, cspace.robot_radius, facecolor='blue', alpha=0.3, edgecolor='blue')
    ax1.add_patch(robot)
    
    # Draw start and goal
    ax1.plot(cspace.start[0], cspace.start[1], 'go', markersize=10, label='Start')
    ax1.plot(cspace.goal[0], cspace.goal[1], 'ro', markersize=10, label='Goal')
    
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: C-Space (with grown obstacles)
    ax2 = axes[1]
    ax2.set_xlim(0, cspace.width)
    ax2.set_ylim(0, cspace.height)
    ax2.set_aspect('equal')
    ax2.set_title('Configuration Space (Grown Obstacles)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    
    # Draw C-Space obstacles
    for obs in cspace.cspace_obstacles:
        if obs['type'] == 'rectangle':
            x, y, w, h = obs['params']
            rect = Rectangle((x, y), w, h, facecolor='red', alpha=0.7, edgecolor='darkred', linewidth=2)
            ax2.add_patch(rect)
        elif obs['type'] == 'circle':
            cx, cy, r = obs['params']
            circle = Circle((cx, cy), r, facecolor='red', alpha=0.7, edgecolor='darkred', linewidth=2)
            ax2.add_patch(circle)
    
    # Draw path in C-Space
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax2.plot(path_x, path_y, 'g-', linewidth=3, label='Planned Path')
    ax2.scatter(path_x, path_y, c='green', s=50, zorder=5)
    
    # Color path segments based on collision
    for i in range(len(path) - 1):
        if cspace.line_in_cspace_obstacle(path[i], path[i+1]):
            ax2.plot([path[i][0], path[i+1][0]], 
                    [path[i][1], path[i+1][1]], 
                    'r-', linewidth=4, alpha=0.5)
    
    # Draw start and goal
    ax2.plot(cspace.start[0], cspace.start[1], 'go', markersize=12, label='Start')
    ax2.plot(cspace.goal[0], cspace.goal[1], 'ro', markersize=12, label='Goal')
    
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('cspace_motion_planning.png')
    print(f"\n✅ Visualization saved as 'cspace_motion_planning.png'")
    
except Exception as e:
    print(f"\n⚠️ Visualization skipped: {e}")

print("\n" + "=" * 70)
print("📌 C-Space Theory (from Planning Algorithms):")
print("=" * 70)
print("""
Configuration Space (C-Space):
    • Each robot configuration = point in C-Space
    • C-obstacles = {q ∈ C | robot at q collides with workspace obstacles}
    • C-free = C \ C-obstacles
    • Path planning = find continuous path in C-free

Minkowski Sum:
    • For convex robot R and obstacle O:
      C-obstacle = O ⊕ R = {o + r | o ∈ O, r ∈ R}
    • Grows obstacle by robot shape
    • Allows planning with point robot in C-Space
""")

# How to use with real motion planning libraries
print("\n🔍 Using with Motion Planning Libraries:")
print("=" * 70)

print("\n📥 OMPL (Open Motion Planning Library):")
print("```python")
print("from ompl import base as ob")
print("from ompl import geometric as og")
print("")
print("# Create state space (C-Space)")
print("space = ob.RealVectorStateSpace(2)")
print("space.setBounds(0, 20)  # x and y bounds")
print("")
print("# Define collision checking")
print("def isStateValid(state):")
print("    # Check if configuration is collision-free")
print("    x = state[0]")
print("    y = state[1]")
print("    # Return True if not in C-obstacle")
print("    return not cspace.point_in_cspace_obstacle((x, y))")
print("")
print("# Set up problem")
print("ss = og.SimpleSetup(space)")
print("ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))")
print("")
print("# Set start and goal")
print("start = ob.State(space)")
print("start[0], start[1] = 1, 1")
print("goal = ob.State(space)")
print("goal[0], goal[1] = 18, 18")
print("ss.setStartAndGoalStates(start, goal)")
print("")
print("# Solve")
print("ss.solve(5.0)")
print("if ss.haveSolution():")
print("    print('Collision-Free path found')")
print("```")

print("\n📦 PythonRobotics:")
print("```python")
print("from PythonRobotics.PathPlanning import Dijkstra")
print("")
print("# Create grid map in C-Space")
print("ox, oy = [], []")
print("for obs in cspace.cspace_obstacles:")
print("    # Convert C-obstacles to grid cells")
print("    # ... (grid discretization)")
print("")
print("dijkstra = Dijkstra(ox, oy, grid_size=0.5, robot_radius=0.0)")
print("rx, ry = dijkstra.planning(1.0, 1.0, 18.0, 18.0)")
print("print('Path found:', len(rx), 'waypoints')")
print("```")

print("\n" + "=" * 70)
print("🎯 SUMMARY")
print("=" * 70)
print(f"✓ C-Space created with {len(cspace.obstacles)} obstacles")
print(f"✓ Minkowski sum applied (robot radius = {cspace.robot_radius}m)")
print(f"✓ Path from {cspace.start} to {cspace.goal}")
print(f"✓ Collision check: {'FREE' if collision_free else 'BLOCKED'}")
print(f"✓ OUTPUT: {'Collision-Free' if collision_free else 'Path in Collision'}")
print("✓ Industry: NVIDIA Autonomous Systems")
print("=" * 70)
