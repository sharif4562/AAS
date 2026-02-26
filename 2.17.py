import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import random
import math

class RRTPlanner:
    """
    Rapidly-exploring Random Tree (RRT) path planning implementation
    Based on algorithms from OMPL (Open Motion Planning Library)
    """
    
    def __init__(self, start, goal, obstacle_list, bounds, 
                 expand_dis=2.0, goal_sample_rate=20, max_iter=500):
        """
        Initialize RRT planner
        
        Args:
            start: [x, y] start position
            goal: [x, y] goal position
            obstacle_list: list of obstacles [x, y, radius]
            bounds: [[x_min, x_max], [y_min, y_max]]
            expand_dis: expansion distance for new nodes
            goal_sample_rate: probability to sample goal directly
            max_iter: maximum iterations
        """
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacles = obstacle_list
        self.min_x, self.max_x = bounds[0]
        self.min_y, self.max_y = bounds[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = [self.start]
        
    def plan(self):
        """
        Execute RRT planning
        
        Returns:
            path: list of [x, y] positions from start to goal
        """
        print("🔄 Running RRT path planning...")
        
        for i in range(self.max_iter):
            # Sample random point (with goal bias)
            if random.randint(0, 100) < self.goal_sample_rate:
                sample = [self.goal.x, self.goal.y]
                print(f"   Iter {i}: Sampling goal directly")
            else:
                sample = self.random_sample()
                print(f"   Iter {i}: Random sample at ({sample[0]:.1f}, {sample[1]:.1f})")
            
            # Find nearest node in tree
            nearest_node = self.nearest_node(sample)
            
            # Steer towards sample
            new_node = self.steer(nearest_node, sample)
            
            # Check if path is collision-free
            if self.collision_free(nearest_node, new_node):
                self.node_list.append(new_node)
                print(f"      ✅ Added node at ({new_node.x:.1f}, {new_node.y:.1f})")
                
                # Check if we reached goal
                dist_to_goal = self.distance(new_node, self.goal)
                if dist_to_goal <= self.expand_dis:
                    print(f"      🎯 Goal reached! Distance: {dist_to_goal:.2f}")
                    final_node = self.steer(new_node, [self.goal.x, self.goal.y])
                    if self.collision_free(new_node, final_node):
                        self.node_list.append(final_node)
                        return self.extract_path(final_node)
            else:
                print(f"      ❌ Path blocked by obstacle")
        
        print("⚠️ Max iterations reached without finding path")
        return None
    
    def random_sample(self):
        """Generate random sample within bounds"""
        x = random.uniform(self.min_x, self.max_x)
        y = random.uniform(self.min_y, self.max_y)
        return [x, y]
    
    def nearest_node(self, sample):
        """Find node in tree closest to sample"""
        distances = [self.distance(node, sample) for node in self.node_list]
        nearest_idx = distances.index(min(distances))
        return self.node_list[nearest_idx]
    
    def steer(self, from_node, to_point):
        """Create new node in direction of sample"""
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < self.expand_dis:
            # Sample is close enough
            new_x, new_y = to_point
        else:
            # Move only expand_dis towards sample
            theta = math.atan2(dy, dx)
            new_x = from_node.x + self.expand_dis * math.cos(theta)
            new_y = from_node.y + self.expand_dis * math.sin(theta)
        
        new_node = Node(new_x, new_y)
        new_node.parent = from_node
        return new_node
    
    def collision_free(self, from_node, to_node):
        """Check if path between nodes is collision-free"""
        for (ox, oy, radius) in self.obstacles:
            # Check if line segment intersects obstacle
            if self.line_collision(from_node, to_node, (ox, oy), radius):
                return False
        return True
    
    def line_collision(self, node1, node2, obstacle_center, radius):
        """Check if line segment between nodes collides with circular obstacle"""
        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y
        ox, oy = obstacle_center
        
        # Vector from node1 to node2
        dx = x2 - x1
        dy = y2 - y1
        fx = x1 - ox
        fy = y1 - oy
        
        a = dx**2 + dy**2
        b = 2 * (fx*dx + fy*dy)
        c = fx**2 + fy**2 - radius**2
        
        discriminant = b*b - 4*a*c
        if discriminant >= 0:
            discriminant = math.sqrt(discriminant)
            t1 = (-b - discriminant) / (2*a)
            t2 = (-b + discriminant) / (2*a)
            
            # Check if collision point is between nodes
            if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                return True
        return False
    
    def distance(self, node, point):
        """Euclidean distance between node and point"""
        return math.sqrt((node.x - point[0])**2 + (node.y - point[1])**2)
    
    def extract_path(self, node):
        """Extract path from start to goal"""
        path = []
        current = node
        while current is not None:
            path.append([current.x, current.y])
            current = current.parent
        return path[::-1]  # Reverse to get start->goal

class Node:
    """RRT Node"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

# Main execution
print("=" * 70)
print("NVIDIA - RRT Path Planning: OMPL Benchmarks")
print("=" * 70)

print("\n🌲 Rapidly-exploring Random Tree (RRT)")
print("   • Sampling-based motion planning algorithm")
print("   • Probabilistically complete")
print("   • Used in OMPL (Open Motion Planning Library)")

# Define environment (simulating OMPL benchmark)
print("\n🗺️  Environment Setup:")
print("   Bounds: X[0, 50], Y[0, 50]")
print("   Start: (5, 5)")
print("   Goal: (45, 45)")

# Obstacles: [x, y, radius]
obstacles = [
    (15, 15, 5),   # Obstacle 1
    (25, 30, 6),   # Obstacle 2
    (35, 20, 4),   # Obstacle 3
    (20, 40, 5),   # Obstacle 4
    (40, 10, 5),   # Obstacle 5
]

print("   Obstacles:")
for i, (ox, oy, r) in enumerate(obstacles):
    print(f"     {i+1}: Center=({ox}, {oy}), Radius={r}")

# Create RRT planner
rrt = RRTPlanner(
    start=[5, 5],
    goal=[45, 45],
    obstacle_list=obstacles,
    bounds=[[0, 50], [0, 50]],
    expand_dis=4.0,
    goal_sample_rate=10,
    max_iter=300
)

# Run RRT planning
print("\n" + "-" * 70)
path = rrt.plan()
print("-" * 70)

# Output result
print("\n" + "=" * 70)
if path is not None:
    print("🎯 OUTPUT: Goal Connected")
    print(f"   Industry: NVIDIA Autonomous Systems")
    print(f"   Path found with {len(path)} waypoints")
    print(f"   Path length: {sum(math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) for i in range(len(path)-1)):.1f} units")
else:
    print("❌ OUTPUT: Path Not Found")
    print("   Industry: NVIDIA Autonomous Systems")
print("=" * 70)

# Show path details
if path:
    print("\n📋 Path Waypoints:")
    print("   " + " → ".join([f"({p[0]:.1f},{p[1]:.1f})" for p in path[:5]]) + 
          (" ..." if len(path) > 5 else ""))
    
    # Statistics
    print(f"\n📊 RRT Statistics:")
    print(f"   • Nodes explored: {len(rrt.node_list)}")
    print(f"   • Path nodes: {len(path)}")
    print(f"   • Success: Connected start to goal")

# Visualize
try:
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: RRT Tree and Path
    ax1 = ax[0]
    
    # Draw obstacles
    for (ox, oy, r) in obstacles:
        circle = Circle((ox, oy), r, color='red', alpha=0.5, label='Obstacle' if ox==15 else "")
        ax1.add_patch(circle)
    
    # Draw RRT tree
    for node in rrt.node_list:
        if node.parent:
            ax1.plot([node.x, node.parent.x], [node.y, node.parent.y], 
                    'b-', linewidth=0.5, alpha=0.3)
    
    # Draw nodes
    node_x = [node.x for node in rrt.node_list]
    node_y = [node.y for node in rrt.node_list]
    ax1.scatter(node_x, node_y, c='blue', s=10, alpha=0.5, label='Tree nodes')
    
    # Draw path if found
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax1.plot(path_x, path_y, 'g-', linewidth=3, label='RRT Path')
        ax1.scatter(path_x, path_y, c='green', s=50, zorder=5)
    
    # Start and goal
    ax1.plot(5, 5, 'go', markersize=15, label='Start', markeredgecolor='white')
    ax1.plot(45, 45, 'ro', markersize=15, label='Goal', markeredgecolor='white')
    
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 50)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('RRT Path Planning')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Exploration progress
    ax2 = ax[1]
    
    # Simulate exploration over iterations
    iterations = list(range(1, len(rrt.node_list)+1))
    distances_to_goal = []
    for node in rrt.node_list:
        dist = math.sqrt((node.x - 45)**2 + (node.y - 45)**2)
        distances_to_goal.append(dist)
    
    ax2.plot(iterations, distances_to_goal, 'b-', linewidth=2)
    ax2.set_xlabel('Nodes Added')
    ax2.set_ylabel('Distance to Goal')
    ax2.set_title('RRT Convergence')
    ax2.grid(True, alpha=0.3)
    
    # Mark when goal connected
    if path:
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.text(len(rrt.node_list), 5, 'Goal Connected', ha='right', color='green')
    
    plt.tight_layout()
    plt.savefig('rrt_path_planning.png')
    print(f"\n✅ Visualization saved as 'rrt_path_planning.png'")
    
except Exception as e:
    print(f"\n⚠️ Visualization skipped: {e}")

print("\n" + "=" * 70)
print("📌 About OMPL (Open Motion Planning Library):")
print("   • Collection of sampling-based motion planning algorithms")
print("   • Includes RRT, RRT*, PRM, EST, KPIECE, and many more")
print("   • Used in ROS MoveIt, V-REP, OpenRAVE")
print("   • Designed for easy integration with collision checkers")
print("=" * 70)

# Reference to OMPL
print("\n📚 OMPL Resources:")
print("   • Website: https://ompl.kavrakilab.org")
print("   • Documentation: https://ompl.kavrakilab.org/classompl_1_1geometric_1_1RRT.html")
print("   • Python bindings: https://ompl.kavrakilab.org/installation.html#pythonBindings")

# How to use actual OMPL with Python
print("\n🔍 Example using OMPL Python bindings:")
print("```python")
print("from ompl import base as ob")
print("from ompl import geometric as og")
print("")
print("# Create state space")
print("space = ob.RealVectorStateSpace(2)")
print("space.setBounds(0, 50)")
print("")
print("# Create problem definition")
print("ss = og.SimpleSetup(space)")
print("ss.setPlanner(og.RRT(ss.getSpaceInformation()))")
print("")
print("# Set start and goal")
print("start = ob.State(space)")
print("start[0], start[1] = 5, 5")
print("goal = ob.State(space)")
print("goal[0], goal[1] = 45, 45")
print("")
print("# Solve")
print("ss.solve(5.0)  # 5 second time limit")
print("if ss.haveSolution():")
print("    print('Goal Connected')")
print("    ss.getSolutionPath().printAsMatrix()")
print("```")
