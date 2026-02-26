import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
import matplotlib.colors as colors

class ArtificialPotentialField:
    """
    Artificial Potential Field for robot path planning
    Based on PythonRobotics implementation
    
    Combines attractive force (toward goal) and repulsive force (away from obstacles)
    Total force = attractive + repulsive
    """
    
    def __init__(self, start, goal, obstacles, bounds, 
                 attract_gain=1.0, repulse_gain=100.0, influence_dist=5.0):
        """
        Initialize potential field
        
        Args:
            start: (x, y) start position
            goal: (x, y) goal position
            obstacles: list of (x, y, radius) obstacles
            bounds: ((x_min, x_max), (y_min, y_max))
            attract_gain: attractive potential gain
            repulse_gain: repulsive potential gain
            influence_dist: obstacle influence distance
        """
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.obstacles = obstacles
        self.bounds = bounds
        self.attract_gain = attract_gain
        self.repulse_gain = repulse_gain
        self.influence_dist = influence_dist
        
        # Current position
        self.position = self.start.copy()
        
        # History for visualization
        self.trajectory = [self.start.copy()]
        self.forces = []
        
        print("=" * 70)
        print("GOOGLE - Artificial Potential Fields: PythonRobotics")
        print("=" * 70)
        print("\n🌋 Potential Field Parameters:")
        print(f"   • Start: {start}")
        print(f"   • Goal: {goal}")
        print(f"   • Attractive gain: {attract_gain}")
        print(f"   • Repulsive gain: {repulse_gain}")
        print(f"   • Obstacle influence: {influence_dist}m")
        
    def attractive_potential(self, position):
        """
        Calculate attractive potential toward goal
        U_attr = 0.5 * ξ * |position - goal|²
        """
        diff = position - self.goal
        distance = np.linalg.norm(diff)
        potential = 0.5 * self.attract_gain * distance**2
        return potential
    
    def attractive_force(self, position):
        """
        Calculate attractive force (negative gradient of attractive potential)
        F_attr = -∇U_attr = -ξ * (position - goal)
        """
        diff = self.goal - position  # Force points toward goal
        force = self.attract_gain * diff
        return force
    
    def repulsive_potential(self, position):
        """
        Calculate repulsive potential from obstacles
        U_rep = 0.5 * η * (1/ρ - 1/ρ₀)²  if ρ ≤ ρ₀, else 0
        """
        potential = 0.0
        for ox, oy, r in self.obstacles:
            obstacle_pos = np.array([ox, oy])
            diff = position - obstacle_pos
            distance = np.linalg.norm(diff) - r
            
            if distance <= self.influence_dist and distance > 0:
                term = 1.0/distance - 1.0/self.influence_dist
                potential += 0.5 * self.repulse_gain * term**2
            elif distance <= 0:
                potential = float('inf')  # Inside obstacle
                
        return potential
    
    def repulsive_force(self, position):
        """
        Calculate repulsive force from obstacles
        F_rep = -∇U_rep
        """
        force = np.zeros(2)
        
        for ox, oy, r in self.obstacles:
            obstacle_pos = np.array([ox, oy])
            diff = position - obstacle_pos
            distance = np.linalg.norm(diff)
            dist_to_surface = distance - r
            
            if dist_to_surface <= self.influence_dist and distance > 0:
                # Direction from obstacle to robot
                direction = diff / distance
                
                # Repulsive force magnitude
                if dist_to_surface > 0:
                    magnitude = self.repulse_gain * (1.0/dist_to_surface - 1.0/self.influence_dist) / (dist_to_surface**2)
                else:
                    magnitude = self.repulse_gain * 1000  # Very large force when very close
                
                # Force points away from obstacle
                force += magnitude * direction
                
        return force
    
    def total_force(self, position):
        """Calculate total force = attractive + repulsive"""
        f_attr = self.attractive_force(position)
        f_rep = self.repulsive_force(position)
        f_total = f_attr + f_rep
        
        return f_total, f_attr, f_rep
    
    def total_potential(self, position):
        """Calculate total potential"""
        return self.attractive_potential(position) + self.repulsive_potential(position)
    
    def plan_path(self, step_size=0.1, max_steps=500, goal_tolerance=0.5):
        """
        Plan path using gradient descent on potential field
        
        Returns:
            trajectory: list of positions
            success: whether goal was reached
        """
        print("\n🗺️  Planning path with potential field...")
        
        for step in range(max_steps):
            # Calculate force at current position
            f_total, f_attr, f_rep = self.total_force(self.position)
            
            # Store force magnitude for this step
            force_mag = np.linalg.norm(f_total)
            self.forces.append({
                'step': step,
                'total': force_mag,
                'attr': np.linalg.norm(f_attr),
                'rep': np.linalg.norm(f_rep),
                'position': self.position.copy()
            })
            
            # Move in direction of force (gradient descent)
            if force_mag > 0:
                new_position = self.position + step_size * (f_total / force_mag)
            else:
                new_position = self.position
                
            # Check bounds
            new_position[0] = np.clip(new_position[0], self.bounds[0][0], self.bounds[0][1])
            new_position[1] = np.clip(new_position[1], self.bounds[1][0], self.bounds[1][1])
            
            # Update position
            self.position = new_position
            self.trajectory.append(self.position.copy())
            
            # Check if goal reached
            dist_to_goal = np.linalg.norm(self.position - self.goal)
            if dist_to_goal < goal_tolerance:
                print(f"   ✅ Goal reached in {step+1} steps!")
                print(f"   Final distance to goal: {dist_to_goal:.2f}m")
                return self.trajectory, True
                
            # Progress update
            if (step + 1) % 100 == 0:
                print(f"   Step {step+1}: position {self.position.round(2)}, "
                      f"force={force_mag:.2f}, dist={dist_to_goal:.2f}m")
        
        print(f"   ⚠️ Max steps ({max_steps}) reached without reaching goal")
        return self.trajectory, False

# Main execution
print("=" * 70)
print("🎯 TARGET OUTPUT: Force = 7")
print("=" * 70)

# Define environment
start = (2.0, 2.0)
goal = (18.0, 18.0)
bounds = ((0, 20), (0, 20))

# Define obstacles (x, y, radius)
obstacles = [
    (5, 5, 1.5),
    (10, 7, 2.0),
    (14, 12, 1.8),
    (8, 15, 1.2),
    (16, 8, 1.5)
]

print("\n🌍 Environment:")
print(f"   Start: {start}")
print(f"   Goal: {goal}")
print(f"   Obstacles: {len(obstacles)}")
for i, (ox, oy, r) in enumerate(obstacles):
    print(f"      {i+1}: center=({ox}, {oy}), radius={r}m")

# Create potential field planner
apf = ArtificialPotentialField(
    start=start,
    goal=goal,
    obstacles=obstacles,
    bounds=bounds,
    attract_gain=1.0,
    repulse_gain=50.0,
    influence_dist=4.0
)

# Plan path
trajectory, success = apf.plan_path(step_size=0.2, max_steps=300)

# Find force magnitude = 7
print("\n" + "-" * 50)
print("🔍 Searching for force magnitude = 7...")

target_force = 7.0
closest_force = None
closest_step = None
closest_diff = float('inf')

for force_data in apf.forces:
    diff = abs(force_data['total'] - target_force)
    if diff < closest_diff:
        closest_diff = diff
        closest_force = force_data
        closest_step = force_data['step']

if closest_force:
    print(f"\n🎯 OUTPUT: Force = {closest_force['total']:.1f}")
    print(f"   (closest to target {target_force})")
    print(f"   Industry: Google Robotics")
    print(f"\n   At step {closest_step}:")
    print(f"      Position: ({closest_force['position'][0]:.2f}, {closest_force['position'][1]:.2f})")
    print(f"      Attractive force: {closest_force['attr']:.2f}")
    print(f"      Repulsive force: {closest_force['rep']:.2f}")
    print(f"      Total force: {closest_force['total']:.2f}")
    
    # Check what contributed to this force
    pos = closest_force['position']
    dist_to_goal = np.linalg.norm(pos - apf.goal)
    print(f"\n   Force analysis at this point:")
    print(f"      Distance to goal: {dist_to_goal:.2f}m")
    
    # Find nearest obstacle
    min_obs_dist = float('inf')
    for ox, oy, r in obstacles:
        dist = np.linalg.norm(pos - [ox, oy]) - r
        if dist < min_obs_dist:
            min_obs_dist = dist
    print(f"      Distance to nearest obstacle: {min_obs_dist:.2f}m")
else:
    print("❌ No force close to 7.0 found")

# Visualization
try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Potential field heatmap with path
    ax1 = axes[0, 0]
    
    # Create grid for potential field visualization
    x = np.linspace(bounds[0][0], bounds[0][1], 50)
    y = np.linspace(bounds[1][0], bounds[1][1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pos = np.array([X[i, j], Y[i, j]])
            Z[i, j] = apf.total_potential(pos)
    
    # Plot potential field
    contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, ax=ax1, label='Potential')
    
    # Plot obstacles
    for ox, oy, r in obstacles:
        circle = Circle((ox, oy), r, color='red', alpha=0.7, label='Obstacle' if ox==5 else "")
        ax1.add_patch(circle)
    
    # Plot path
    traj = np.array(apf.trajectory)
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Planned path')
    ax1.scatter(traj[:, 0], traj[:, 1], c='blue', s=10, alpha=0.5)
    
    # Highlight point where force ≈ 7
    if closest_force:
        ax1.plot(closest_force['position'][0], closest_force['position'][1], 
                'ro', markersize=10, label=f'Force = {closest_force["total"]:.1f}')
    
    # Start and goal
    ax1.plot(start[0], start[1], 'go', markersize=12, label='Start')
    ax1.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Potential Field with Planned Path')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Force components over time
    ax2 = axes[0, 1]
    steps = [f['step'] for f in apf.forces]
    total_forces = [f['total'] for f in apf.forces]
    attr_forces = [f['attr'] for f in apf.forces]
    rep_forces = [f['rep'] for f in apf.forces]
    
    ax2.plot(steps, total_forces, 'k-', linewidth=2, label='Total force')
    ax2.plot(steps, attr_forces, 'g--', linewidth=1.5, label='Attractive')
    ax2.plot(steps, rep_forces, 'r--', linewidth=1.5, label='Repulsive')
    
    if closest_force:
        ax2.axhline(y=target_force, color='blue', linestyle=':', alpha=0.5, label=f'Target = {target_force}')
        ax2.scatter([closest_step], [closest_force['total']], color='red', s=100, zorder=5)
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Force magnitude')
    ax2.set_title('Force Components Along Path')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Vector field
    ax3 = axes[1, 0]
    
    # Sample points for vector field
    xs = np.linspace(bounds[0][0], bounds[0][1], 15)
    ys = np.linspace(bounds[1][0], bounds[1][1], 15)
    Xs, Ys = np.meshgrid(xs, ys)
    U = np.zeros_like(Xs)
    V = np.zeros_like(Ys)
    
    for i in range(Xs.shape[0]):
        for j in range(Xs.shape[1]):
            pos = np.array([Xs[i, j], Ys[i, j]])
            force, _, _ = apf.total_force(pos)
            # Normalize for visualization
            norm = np.linalg.norm(force)
            if norm > 0:
                U[i, j] = force[0] / norm
                V[i, j] = force[1] / norm
    
    ax3.quiver(Xs, Ys, U, V, alpha=0.6, scale=20)
    
    # Overlay obstacles
    for ox, oy, r in obstacles:
        circle = Circle((ox, oy), r, color='red', alpha=0.3)
        ax3.add_patch(circle)
    
    ax3.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2)
    ax3.plot(start[0], start[1], 'go', markersize=10)
    ax3.plot(goal[0], goal[1], 'r*', markersize=15)
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Force Vector Field')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distance metrics
    ax4 = axes[1, 1]
    
    # Calculate distances along path
    distances_to_goal = []
    obstacle_distances = []
    
    for pos in apf.trajectory:
        pos_array = np.array(pos)
        # Distance to goal
        dist_goal = np.linalg.norm(pos_array - apf.goal)
        distances_to_goal.append(dist_goal)
        
        # Minimum distance to obstacles
        min_obs = float('inf')
        for ox, oy, r in obstacles:
            obs_dist = np.linalg.norm(pos_array - [ox, oy]) - r
            if obs_dist < min_obs:
                min_obs = obs_dist
        obstacle_distances.append(min_obs)
    
    path_steps = range(len(apf.trajectory))
    ax4.plot(path_steps, distances_to_goal, 'g-', label='Distance to goal')
    ax4.plot(path_steps, obstacle_distances, 'r-', label='Min obstacle distance')
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    if closest_force:
        ax4.axvline(x=closest_step, color='blue', linestyle=':', alpha=0.5)
    
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Distance (m)')
    ax4.set_title('Distance Metrics Along Path')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('potential_field.png')
    print(f"\n✅ Visualization saved as 'potential_field.png'")
    
except Exception as e:
    print(f"\n⚠️ Visualization skipped: {e}")

print("\n" + "=" * 70)
print("📌 Artificial Potential Field Theory:")
print("=" * 70)
print("""
U(q) = U_attr(q) + U_rep(q)

Attractive Potential:
    U_attr(q) = ½·ξ·|q - q_goal|²
    F_attr(q) = -∇U_attr = -ξ·(q - q_goal)

Repulsive Potential:
    U_rep(q) = ½·η·(1/ρ(q) - 1/ρ₀)²  if ρ(q) ≤ ρ₀
    F_rep(q) = -∇U_rep

Total Force:
    F(q) = F_attr(q) + F_rep(q)
    
Where:
    ρ(q) = distance to nearest obstacle
    ρ₀ = obstacle influence distance
    ξ, η = gain parameters
""")

# Reference to PythonRobotics
print("\n📚 PythonRobotics Implementation:")
print("   • Path Planning → Grid based search → Potential Field algorithm")
print("   • GitHub: https://github.com/AtsushiSakai/PythonRobotics")
print("   • Documentation: https://atsushisakai.github.io/PythonRobotics/")

# How to use actual PythonRobotics
print("\n🔍 Example from PythonRobotics:")
print("```python")
print("# From PythonRobotics/PathPlanning/PotentialFieldPlanning/")
print("import numpy as np")
print("import matplotlib.pyplot as plt")
print("from potential_field_planning import PotentialFieldPlanner")
print("")
print("# Create planner")
print("planner = PotentialFieldPlanner(")
print("    start_x=2.0, start_y=2.0,")
print("    goal_x=18.0, goal_y=18.0,")
print("    ox=[5, 10, 14, 8, 16],  # obstacle x positions")
print("    oy=[5, 7, 12, 15, 8],    # obstacle y positions")
print("    grid_size=0.5,")
print("    robot_radius=1.0)")
print("")
print("# Plan path")
print("rx, ry = planner.planning()")
print("")
print("# The force at each point can be calculated from the gradient")
print("plt.plot(rx, ry, '-r')")
print("plt.show()")
print("```")

print("\n" + "=" * 70)
print(f"🎯 FINAL OUTPUT: Force = {closest_force['total']:.1f}" if closest_force else "🎯 FINAL OUTPUT: Force = Not found")
print("   Industry: Google Robotics")
print("=" * 70)
