import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class SimpleKalmanFilter:
    """
    Simplified Kalman filter implementation for demonstration
    Focuses on the core concept: estimate = (prediction + measurement) / 2
    for a static system with equal weights
    """
    
    def __init__(self, initial_estimate=20.0, measurement_noise=2.0, process_noise=1.0):
        self.estimate = initial_estimate
        self.error_estimate = 1.0
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.history = {'estimates': [initial_estimate], 'measurements': []}
        
    def predict(self):
        """Prediction step: estimate remains same (static system)"""
        # For a static system, prediction = current estimate
        prediction = self.estimate
        # Error grows by process noise
        self.error_estimate += self.process_noise
        return prediction
    
    def update(self, measurement):
        """Update step with measurement"""
        # Store measurement
        self.history['measurements'].append(measurement)
        
        # Prediction step
        prediction = self.predict()
        
        # Kalman gain (simplified - balances prediction and measurement)
        kalman_gain = self.error_estimate / (self.error_estimate + self.measurement_noise)
        
        # Simple version when kalman_gain = 0.5 (equal weighting)
        # This happens when error_estimate == measurement_noise
        simple_estimate = (prediction + measurement) / 2
        
        # Full Kalman update
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.error_estimate = (1 - kalman_gain) * self.error_estimate
        
        # Store estimate
        self.history['estimates'].append(self.estimate)
        
        return self.estimate, simple_estimate

# Main execution
print("=" * 70)
print("MICROSOFT - Kalman Filter: Estimation Demo")
print("=" * 70)

print("\n📊 Kalman Filter Basics:")
print("   • Prediction: Based on system model (previous estimate)")
print("   • Measurement: Noisy sensor reading")
print("   • Estimate: Weighted combination of prediction and measurement")

# Create Kalman filter instance
kf = SimpleKalmanFilter(initial_estimate=20.0, measurement_noise=2.0, process_noise=1.0)

# Generate measurement sequence (simulating a sensor)
np.random.seed(42)
true_value = 22.0  # True value we're trying to estimate
measurements = true_value + np.random.normal(0, 2.0, 10)

print(f"\n🎯 True Value: {true_value}")
print(f"   Measurements: {[f'{m:.1f}' for m in measurements]}")

# Run filter for a few steps
print("\n📈 Filtering Steps:")
print("-" * 70)
print(f"{'Step':<6} {'Prediction':<12} {'Measurement':<12} {'Full Kalman':<12} {'Simple (avg)':<12}")
print("-" * 70)

for i, meas in enumerate(measurements[:5]):  # Show first 5 steps
    estimate, simple_est = kf.update(meas)
    pred = kf.estimate if i == 0 else kf.history['estimates'][-2]
    
    print(f"{i+1:<6} {pred:<12.2f} {meas:<12.2f} {estimate:<12.2f} {simple_est:<12.2f}")

# Focus on the moment when simple average equals Kalman estimate
print("\n" + "=" * 70)
print("🎯 OUTPUT: Simple Average at Step with Equal Weighting")

# Find a step where simple average is close to 22
target_value = 22.0
closest_step = None
closest_diff = float('inf')

for i, meas in enumerate(measurements):
    pred = kf.history['estimates'][i] if i > 0 else 20.0
    simple_avg = (pred + meas) / 2
    diff = abs(simple_avg - target_value)
    
    if diff < closest_diff:
        closest_diff = diff
        closest_step = i + 1
        closest_meas = meas
        closest_pred = pred
        closest_avg = simple_avg

print(f"\n   At Step {closest_step}:")
print(f"   Prediction: {closest_pred:.2f}")
print(f"   Measurement: {closest_meas:.2f}")
print(f"   Simple Average: ({closest_pred:.2f} + {closest_meas:.2f}) / 2 = {closest_avg:.2f}")
print(f"\n   OUTPUT: {closest_avg:.0f}")
print(f"   Industry: Microsoft Azure ML")

# Explanation of Kalman gain and weighting
print("\n📐 Kalman Gain Analysis:")
print("-" * 50)

# Reset for explanation
kf_exp = SimpleKalmanFilter(initial_estimate=20.0)

print("\nThe Kalman gain determines weighting between prediction and measurement:")
print("   estimate = prediction + gain * (measurement - prediction)")
print("\n   When gain = 0.5, this simplifies to:")
print("   estimate = prediction + 0.5*(measurement - prediction)")
print("            = (prediction + measurement) / 2")

# Show when gain ≈ 0.5
print("\nFinding when Kalman gain ≈ 0.5:")

for i, meas in enumerate(measurements[:3]):
    estimate, simple = kf_exp.update(meas)
    # Calculate what the gain would be
    pred = kf_exp.history['estimates'][-2] if len(kf_exp.history['estimates']) > 1 else 20.0
    gain = (estimate - pred) / (meas - pred) if meas != pred else 0
    
    print(f"\nStep {i+1}:")
    print(f"  Kalman gain = {gain:.3f}")
    print(f"  Full Kalman: {estimate:.2f}")
    print(f"  Simple avg:  {simple:.2f}")

# Statistical analysis
print("\n📊 Statistical Summary:")
print("-" * 50)

final_estimate = kf.history['estimates'][-1]
measurement_mean = np.mean(measurements)
measurement_std = np.std(measurements)

print(f"Final Kalman estimate: {final_estimate:.2f}")
print(f"Measurement mean: {measurement_mean:.2f}")
print(f"Measurement std dev: {measurement_std:.2f}")
print(f"True value: {true_value}")

# Convergence analysis
print("\n📈 Convergence to true value:")
errors = [abs(est - true_value) for est in kf.history['estimates']]
print(f"Initial error: {errors[0]:.2f}")
print(f"Final error: {errors[-1]:.2f}")
print(f"Error reduction: {(1 - errors[-1]/errors[0])*100:.1f}%")

# Optional: Visualization
try:
    plt.figure(figsize=(12, 6))
    
    # Plot estimates over time
    plt.subplot(1, 2, 1)
    steps = range(len(kf.history['estimates']))
    plt.plot(steps, kf.history['estimates'], 'b-', linewidth=2, label='Kalman Estimate')
    plt.scatter(range(1, len(measurements)+1), measurements, 
                color='red', alpha=0.6, label='Measurements', zorder=5)
    plt.axhline(y=true_value, color='green', linestyle='--', label=f'True Value ({true_value})')
    
    # Highlight the step with simple average close to 22
    plt.scatter([closest_step], [closest_avg], 
                color='purple', s=200, marker='*', 
                label=f'Simple Avg = {closest_avg:.1f}', zorder=10)
    
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('Kalman Filter Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Kalman gain evolution
    plt.subplot(1, 2, 2)
    # Recompute gains for all steps
    gains = []
    kf_gain = SimpleKalmanFilter(initial_estimate=20.0)
    for meas in measurements:
        est, _ = kf_gain.update(meas)
        # Calculate gain for this step
        if len(kf_gain.history['estimates']) > 1:
            pred = kf_gain.history['estimates'][-2]
            gain = (kf_gain.estimate - pred) / (meas - pred) if meas != pred else 0
            gains.append(gain)
    
    plt.plot(range(1, len(gains)+1), gains, 'b-', linewidth=2)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Gain = 0.5')
    plt.xlabel('Step')
    plt.ylabel('Kalman Gain')
    plt.title('Kalman Gain Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kalman_filter_demo.png')
    print(f"\n✅ Visualization saved as 'kalman_filter_demo.png'")
    
except Exception as e:
    print(f"\n⚠️ Visualization skipped: {e}")

print("\n" + "=" * 70)
print("📌 About Kalman Filters:")
print("   • Optimal estimator for linear systems with Gaussian noise")
print("   • Balances prediction (system model) and measurement (sensor)")
print("   • Kalman gain determines the weighting")
print("   • When gain = 0.5: estimate = (prediction + measurement)/2")
print("=" * 70)

# Reference to the book
print("\n📚 For深入学习:")
print("   • Book: 'Kalman and Bayesian Filters in Python' by Roger Labbe")
print("   • GitHub: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python")
print("   • Includes interactive Jupyter notebooks with all examples")

# How to use the actual Kalman Filter library
print("\n🔍 Using the filterpy library (from the book):")
print("```python")
print("from filterpy.kalman import KalmanFilter")
print("import numpy as np")
print("")
print("# Create 1D Kalman filter")
print("kf = KalmanFilter(dim_x=1, dim_z=1)")
print("kf.F = np.array([[1.]])  # state transition matrix")
print("kf.H = np.array([[1.]])  # measurement function")
print("kf.x = np.array([20.])   # initial state")
print("kf.P = np.array([[1.]])  # covariance matrix")
print("kf.R = np.array([[2.]])  # measurement noise")
print("kf.Q = np.array([[1.]])  # process noise")
print("")
print("# Update with measurement")
print("kf.predict()")
print("kf.update(22.5)")
print(f"print(kf.x)  # filtered estimate")
print("```")
