import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class EuRoCSensorFusion:
    """
    Simulates IMU and Vision sensor fusion for EuRoC MAV dataset
    Combines orientation estimates from both sensors
    """
    
    def __init__(self):
        self.imu_data = None
        self.vision_data = None
        self.fused_data = None
        
    def generate_sample_data(self, n_samples=100, true_orientation=30.0, noise_level=2.0):
        """
        Generate synthetic IMU and Vision orientation data
        Based on EuRoC MAV characteristics
        """
        np.random.seed(42)
        
        # True orientation (constant for simplicity)
        true_theta = np.ones(n_samples) * true_orientation
        
        # IMU: high frequency, drifts over time
        time = np.linspace(0, 10, n_samples)
        imu_drift = 0.1 * time  # Gradual drift
        imu_noise = np.random.normal(0, noise_level, n_samples)
        self.imu_data = true_theta + imu_drift + imu_noise
        
        # Vision: lower frequency, more accurate but occasional outliers
        vision_samples = np.zeros(n_samples)
        # Update vision every 5 samples (simulating lower rate)
        for i in range(0, n_samples, 5):
            vision_samples[i:i+5] = true_theta[i] + np.random.normal(0, noise_level/2, 1)
        # Add occasional outlier
        vision_samples[30:35] += 15  # Simulate vision tracking error
        self.vision_data = vision_samples
        
        return time, self.imu_data, self.vision_data
    
    def fuse_sensors(self):
        """
        Fuse IMU and Vision data using simple averaging
        Formula: fused = (imu + vision) / 2
        """
        if self.imu_data is None or self.vision_data is None:
            raise ValueError("Sensor data not available")
        
        self.fused_data = (self.imu_data + self.vision_data) / 2
        return self.fused_data
    
    def get_orientation_at_time(self, time_idx):
        """Get fused orientation at specific time index"""
        if self.fused_data is None:
            self.fuse_sensors()
        return self.fused_data[time_idx]

# Main execution
print("=" * 70)
print("APPLE - iOS Robotics: EuRoC MAV Sensor Fusion")
print("=" * 70)

# Initialize sensor fusion system
fusion = EuRoCSensorFusion()

# Generate sample data (simulating EuRoC MAV dataset)
print("\n📡 EuRoC MAV Dataset Simulation")
print("   Sensors: IMU (200 Hz) + Vision (20 Hz)")
print("   Environment: Industrial hall, Vicon ground truth")

# Generate 100 samples (10 seconds at 10Hz effective rate)
time, imu_data, vision_data = fusion.generate_sample_data(
    n_samples=100, 
    true_orientation=30.0,  # True orientation is 30 degrees
    noise_level=2.0
)

print(f"\n📊 Sensor Readings (first 5 samples):")
print(f"{'Sample':<8} {'Time (s)':<10} {'IMU (°)':<10} {'Vision (°)':<10}")
print("-" * 40)
for i in range(5):
    print(f"{i:<8} {time[i]:<10.2f} {imu_data[i]:<10.2f} {vision_data[i]:<10.2f}")

# Perform sensor fusion
fused_data = fusion.fuse_sensors()

print("\n🔄 Sensor Fusion:")
print(f"   Formula: fused = (imu + vision) / 2")
print(f"   Applied to all {len(time)} samples")

# Get the fused orientation at a specific time (sample 50)
target_sample = 50
fused_orientation = fusion.get_orientation_at_time(target_sample)

print(f"\n🎯 OUTPUT: {fused_orientation:.1f}°")
print(f"   Industry: iOS Robotics")
print(f"\n   At t = {time[target_sample]:.1f}s:")
print(f"   • IMU reading: {imu_data[target_sample]:.1f}°")
print(f"   • Vision reading: {vision_data[target_sample]:.1f}°")
print(f"   • Fused: ({imu_data[target_sample]:.1f} + {vision_data[target_sample]:.1f}) / 2 = {fused_orientation:.1f}°")

# Analysis of fusion benefits
print("\n📈 Fusion Performance Analysis:")
print("-" * 50)

# Calculate errors relative to true orientation (30°)
true_orientation = 30.0
imu_error = np.abs(imu_data - true_orientation).mean()
vision_error = np.abs(vision_data - true_orientation).mean()
fused_error = np.abs(fused_data - true_orientation).mean()

print(f"   Mean Absolute Error:")
print(f"   • IMU only: {imu_error:.2f}°")
print(f"   • Vision only: {vision_error:.2f}°")
print(f"   • Fused: {fused_error:.2f}°")
print(f"   • Improvement: {(1 - fused_error/imu_error)*100:.1f}% vs IMU")

# Show robustness to vision outliers
print(f"\n   Robustness Check (at vision outlier):")
outlier_idx = 32  # Where vision has error
print(f"   • At t={time[outlier_idx]:.1f}s:")
print(f"     - Vision outlier: {vision_data[outlier_idx]:.1f}°")
print(f"     - IMU reading: {imu_data[outlier_idx]:.1f}°")
print(f"     - Fused result: {fused_data[outlier_idx]:.1f}°")
print(f"     - Fusion reduced outlier impact by {(vision_data[outlier_idx] - fused_data[outlier_idx]):.1f}°")

# Optional: Visualize the fusion
try:
    plt.figure(figsize=(12, 8))
    
    # Plot 1: All sensor data
    plt.subplot(2, 1, 1)
    plt.plot(time, imu_data, 'b-', alpha=0.7, label='IMU (noisy, drifting)', linewidth=1)
    plt.plot(time, vision_data, 'g-', alpha=0.7, label='Vision (sparse, accurate)', linewidth=1)
    plt.plot(time, fused_data, 'r-', linewidth=2, label='Fused (averaged)')
    plt.axhline(y=true_orientation, color='k', linestyle='--', label='True orientation')
    plt.axvline(x=time[target_sample], color='purple', linestyle=':', alpha=0.5)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Orientation (degrees)')
    plt.title('EuRoC MAV Sensor Fusion: IMU + Vision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Errors comparison
    plt.subplot(2, 1, 2)
    plt.plot(time, imu_data - true_orientation, 'b-', alpha=0.7, label='IMU error', linewidth=1)
    plt.plot(time, vision_data - true_orientation, 'g-', alpha=0.7, label='Vision error', linewidth=1)
    plt.plot(time, fused_data - true_orientation, 'r-', linewidth=2, label='Fused error')
    plt.axhline(y=0, color='k', linestyle='--')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Error (degrees)')
    plt.title('Estimation Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('euroc_sensor_fusion.png')
    print(f"\n✅ Visualization saved as 'euroc_sensor_fusion.png'")
    
except Exception as e:
    print(f"\n⚠️ Visualization skipped: {e}")

# Statistical summary
print("\n📊 Statistical Summary (all samples):")
print("-" * 50)
print(f"{'Statistic':<20} {'IMU':>10} {'Vision':>10} {'Fused':>10}")
print("-" * 50)
print(f"{'Mean (°):':<20} {imu_data.mean():>10.2f} {np.mean(vision_data[vision_data>0]):>10.2f} {fused_data.mean():>10.2f}")
print(f"{'Std Dev (°):':<20} {imu_data.std():>10.2f} {vision_data.std():>10.2f} {fused_data.std():>10.2f}")
print(f"{'Min (°):':<20} {imu_data.min():>10.2f} {vision_data.min():>10.2f} {fused_data.min():>10.2f}")
print(f"{'Max (°):':<20} {imu_data.max():>10.2f} {vision_data.max():>10.2f} {fused_data.max():>10.2f}")

print("\n" + "=" * 70)
print("📌 About EuRoC MAV Dataset:")
print("   • Micro Aerial Vehicle flights in industrial environments")
print("   • Ground truth from Vicon motion capture")
print("   • Stereo images 20 Hz, IMU 200 Hz")
print("   • Used for visual-inertial odometry research")
print("=" * 70)

# Show how to access real EuRoC data
print("\n🔍 To use real EuRoC MAV data:")
print("   1. Visit: https://ethz-asl.github.io/datasets/")
print("   2. Download 'EuRoC MAV Dataset' sequences")
print("   3. Use provided MATLAB/Python tools")
print("\n   Example real data access:")
print("   ```python")
print("   # After downloading the ASL dataset format")
print("   import h5py")
print("   with h5py.File('V1_01_easy.hdf5', 'r') as f:")
print("       imu_data = f['davis/imu/data'][:]")
print("       images = f['davis/left/image_raw'][:]")
print("   ```")
