import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Define a synthetic optimization function (convex, with minimum at x=3)
# f(x) = (x - 3)^2 + 5
# This is a simple parabola with minimum at x=3, f(x)=5
def objective_function(x):
    """Example objective function: f(x) = (x-3)^2 + 5"""
    return (x - 3)**2 + 5

def analytical_derivative(x):
    """Analytical derivative: f'(x) = 2*(x-3)"""
    return 2 * (x - 3)

def numerical_derivative(x, h=1e-5):
    """Numerical approximation of derivative using central difference"""
    return (objective_function(x + h) - objective_function(x - h)) / (2 * h)

# Step 1: Compute derivative at a few points
print("=" * 60)
print("JPMorgan - AI Ops Optimization")
print("=" * 60)
print("\n1. Computing Derivatives:")

test_points = [0, 2, 3, 4, 6]
for x in test_points:
    analytical = analytical_derivative(x)
    numerical = numerical_derivative(x)
    print(f"   f'({x}) = {analytical:.2f} (analytical), {numerical:.2f} (numerical)")

# Step 2: Find the minimum
# Method 1: Analytical - set derivative to zero
# f'(x) = 2x - 6 = 0 → x = 3
analytical_min_x = 3
analytical_min_value = objective_function(analytical_min_x)

# Method 2: Numerical optimization using SciPy
result = minimize_scalar(objective_function, bounds=(0, 6), method='bounded')
numerical_min_x = result.x
numerical_min_value = result.fun

print(f"\n2. Finding Minimum:")
print(f"   Analytical minimum: x = {analytical_min_x}, f(x) = {analytical_min_value}")
print(f"   Numerical minimum: x = {numerical_min_x:.6f}, f(x) = {numerical_min_value:.6f}")
print(f"\n3. Output: x = {analytical_min_x}")
print(f"   Industry: AI Ops Optimization")

# Optional: Visualize the function and its minimum
x_vals = np.linspace(0, 6, 100)
y_vals = objective_function(x_vals)
derivative_vals = 2 * (x_vals - 3)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = (x-3)² + 5')
plt.plot(analytical_min_x, analytical_min_value, 'ro', markersize=10, label='Minimum (x=3)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Objective Function')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_vals, derivative_vals, 'g-', linewidth=2, label="f'(x) = 2(x-3)")
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.plot(analytical_min_x, 0, 'ro', markersize=10)
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Derivative (Zero at Minimum)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('optimization_result.png')
plt.show()
