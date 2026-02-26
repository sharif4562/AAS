import numpy as np
import matplotlib.pyplot as plt
import heapq
from matplotlib import colors

class GridPathPlanner:
    """
    Grid-based path planning using Dijkstra/A* algorithm
    Finds minimum cost path between start and goal
    """
    
    def __init__(self, grid_size=(10, 10), obstacle_density=0.2):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size)
        self.cost_map = None
        self.paths = {}
        
    def create_sample_grid(self):
        """Create a sample grid with obstacles and traversability costs"""
        np.random.seed(42)
        
        # Initialize with random costs (1 = normal terrain, higher = harder to traverse)
        self.grid = np.random.choice([1, 2, 3, 5], size=self.grid_size, p=[0.6, 0.2, 0.15, 0.05])
        
        # Add obstacles (infinite cost)
        n_obstacles = int(self.grid_size[0] * self.grid_size[1] * 0.15)
        for _ in range(n_obstacles):
            x, y = np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])
            self.grid[x, y] = np.inf
        
        return self.grid
    
    def define_paths(self):
        """
        Define multiple candidate paths between start and goal
        Paths A, B, C with different costs
        """
        start = (1, 1)
        goal = (8, 8)
        
        # Path A: Direct but through high-cost areas
        path_a = [
            (1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8)
        ]
        
        # Path B: Slightly longer but avoids high-cost areas
        path_b = [
            (1,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,8)
        ]
        
        # Path C: Longer route avoiding all obstacles
        path_c = [
            (1,1), (1,3), (2,4), (3,5), (4,6), (5,7), (6,8), (8,8)
        ]
        
        self.paths = {
            'A': path_a,
            'B': path_b,
            'C': path_c
        }
        
        return self.paths
    
    def calculate_path_cost(self, path):
        """Calculate total cost of a path based on grid cell costs"""
        total_cost = 0
        for (x, y) in path:
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                if np.isinf(self.grid[x, y]):
                    return float('inf')  # Path through obstacle
                total_cost += self.grid[x, y]
        return total_cost
    
    def dijkstra_shortest_path(self, start, goal):
        """
        Dijkstra's algorithm to find true minimum cost path
        Used for verification
        """
        rows, cols = self.grid_size
        distances = { (i,j): float('inf') for i in range(rows) for j in range(cols) }
        distances[start] = 0
        previous = { (i,j): None for i in range(rows) for j in range(cols) }
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == goal:
                break
                
            # Check 4-directional neighbors
            x, y = current
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            
            for nx, ny in neighbors:
                if 0 <= nx < rows and 0 <= ny < cols:
                    if np.isinf(self.grid[nx, ny]):
                        continue  # Skip obstacles
                    
                    new_dist = current_dist + self.grid[nx, ny]
                    if new_dist < distances[(nx, ny)]:
                        distances[(nx, ny)] = new_dist
                        previous[(nx, ny)] = current
                        heapq.heappush(pq, (new_dist, (nx, ny)))
        
        # Reconstruct path
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        return path if path[0] == start else None

# Main execution
print("=" * 70)
print("AMAZON - Path Planning: Minimum Cost Path Selection")
print("=" * 70)

# Initialize grid and planner
planner = GridPathPlanner(grid_size=(10, 10))

# Create sample grid with traversability costs
grid = planner.create_sample_grid()
print("\n🗺️  Warehouse Grid Map (10x10)")
print("   Cell values represent traversal cost:")
print("   1 = Free space (low cost)")
print("   2-5 = Increasing difficulty")
print("   inf = Obstacle (impassable)")

# Display grid with costs
print("\n📊 Grid Cost Map:")
print("-" * 50)
print("    " + " ".join([f"{i:3}" for i in range(10)]))
for i in range(10):
    row_str = f"{i:2}  "
    for j in range(10):
        val = grid[i, j]
        if np.isinf(val):
            row_str += "  # "
        else:
            row_str += f"{int(val):3} "
    print(row_str)

# Define candidate paths
paths = planner.define_paths()
print("\n📌 Candidate Paths (Start: (1,1), Goal: (8,8)):")

# Calculate and display costs for each path
path_costs = {}
print("-" * 50)
for path_name, path in paths.items():
    cost = planner.calculate_path_cost(path)
    path_costs[path_name] = cost
    
    if np.isinf(cost):
        cost_str = "INF (blocked)"
    else:
        cost_str = f"{cost:.1f}"
    
    print(f"Path {path_name}: {len(path)} steps, Cost = {cost_str}")
    
    # Show path trajectory
    path_grid = np.full_like(grid, '.', dtype=str)
    for (x, y) in path:
        if 0 <= x < 10 and 0 <= y < 10:
            path_grid[x, y] = path_name
    
    path_str = "   Path: "
    for step in path:
        path_str += f"({step[0]},{step[1]}) "
    print(path_str)

# Find minimum cost path
valid_paths = {name: cost for name, cost in path_costs.items() 
               if not np.isinf(cost)}

if valid_paths:
    min_cost_path = min(valid_paths, key=valid_paths.get)
    min_cost = valid_paths[min_cost_path]
    
    print("\n" + "=" * 70)
    print(f"🎯 OUTPUT: Path {min_cost_path}")
    print(f"   Minimum Cost: {min_cost:.1f}")
    print(f"   Industry: Amazon Fulfillment Robots")
    print("=" * 70)
    
    # Show path details
    print(f"\n📋 Selected Path {min_cost_path} Details:")
    print(f"   Number of steps: {len(paths[min_cost_path])}")
    print(f"   Average cost per step: {min_cost/len(paths[min_cost_path]):.2f}")
    
    # Compare with Dijkstra's optimal path
    optimal_path = planner.dijkstra_shortest_path((1,1), (8,8))
    if optimal_path:
        optimal_cost = planner.calculate_path_cost(optimal_path)
        print(f"\n🔍 Verification with Dijkstra's algorithm:")
        print(f"   True optimal path cost: {optimal_cost:.1f}")
        if abs(optimal_cost - min_cost) < 0.1:
            print(f"   ✅ Path {min_cost_path} is optimal!")
        else:
            print(f"   ⚠️ Path {min_cost_path} is {min_cost - optimal_cost:.1f} higher than optimal")
else:
    print("\n❌ No valid paths found!")

# Visualize the grid and paths
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Cost map with paths
    ax1 = axes[0]
    
    # Create masked array for obstacles
    masked_grid = np.ma.masked_where(np.isinf(grid), grid)
    cmap = plt.cm.YlOrRd
    cmap.set_bad(color='black', alpha=0.8)  # Obstacles in black
    
    im = ax1.imshow(masked_grid, cmap=cmap, interpolation='nearest', 
                    vmin=1, vmax=5)
    
    # Plot all paths
    colors_map = {'A': 'blue', 'B': 'green', 'C': 'purple'}
    for path_name, path in paths.items():
        path_array = np.array(path)
        if len(path_array) > 0:
            ax1.plot(path_array[:, 1], path_array[:, 0], 
                    color=colors_map[path_name], linewidth=2, 
                    label=f'Path {path_name} (cost={path_costs[path_name]:.1f})',
                    marker='o', markersize=4)
    
    # Highlight start and goal
    ax1.plot(1, 1, 'go', markersize=12, label='Start', markeredgecolor='white')
    ax1.plot(8, 8, 'ro', markersize=12, label='Goal', markeredgecolor='white')
    
    ax1.set_title('Warehouse Grid: Candidate Paths')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.colorbar(im, ax=ax1, label='Traversal Cost')
    
    # Plot 2: Cost comparison bar chart
    ax2 = axes[1]
    
    names = list(path_costs.keys())
    costs = [path_costs[n] for n in names]
    colors_bar = ['blue' if not np.isinf(c) else 'gray' for c in costs]
    
    bars = ax2.bar(names, [c if not np.isinf(c) else 0 for c in costs], 
                   color=colors_bar, alpha=0.7)
    
    # Highlight minimum cost path
    if valid_paths:
        min_idx = names.index(min_cost_path)
        bars[min_idx].set_color('green')
        bars[min_idx].set_alpha(1.0)
    
    ax2.set_ylabel('Total Path Cost')
    ax2.set_title('Path Cost Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add cost labels on bars
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        if not np.isinf(cost):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{cost:.1f}', ha='center', va='bottom')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2, 1,
                    'Blocked', ha='center', va='bottom', rotation=90)
    
    plt.tight_layout()
    plt.savefig('warehouse_path_planning.png')
    print(f"\n✅ Visualization saved as 'warehouse_path_planning.png'")
    
except Exception as e:
    print(f"\n⚠️ Visualization skipped: {e}")

print("\n" + "=" * 70)
print("📌 About Path Planning in Warehouses:")
print("   • Grid cells represent traversability cost")
print("   • Obstacles (shelves, other robots) have infinite cost")
print("   • Goal: Find path minimizing total traversal cost")
print("   • Used in Amazon fulfillment centers for robot routing")
print("=" * 70)

# Reference to PythonRobotics
print("\n📚 Reference: PythonRobotics Library")
print("   • GitHub: https://github.com/AtsushiSakai/PythonRobotics")
print("   • Includes implementations of:")
print("     - Dijkstra's algorithm")
print("     - A* search")
print("     - D* and D* Lite for dynamic replanning")
print("     - Many other robotics algorithms")

# How to use actual PythonRobotics for path planning
print("\n🔍 Example using PythonRobotics:")
print("```python")
print("# From PythonRobotics/PathPlanning/Dijkstra/dijkstra.py")
print("import numpy as np")
print("import matplotlib.pyplot as plt")
print("")
print("# Create obstacle map")
print("ox, oy = [], []")
print("for i in range(60):  # obstacles")
print("    ox.append(i)")
print("    oy.append(0.0)")
print("# Run Dijkstra")
print("from dijkstra import Dijkstra")
print("dijkstra = Dijkstra(ox, oy, grid_size=1.0, robot_radius=1.0)")
print("rx, ry = dijkstra.planning(10.0, 10.0, 50.0, 50.0)")
print("plt.plot(rx, ry, '-r')")
print("plt.show()")
print("```")
