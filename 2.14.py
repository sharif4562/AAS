import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

class TUMRGBDSLAM:
    """
    Simulates Visual SLAM pose estimation using TUM RGB-D dataset format
    TUM dataset provides ground truth and allows estimating camera trajectories
    """
    
    def __init__(self):
        self.poses = []
        self.timestamps = []
        
    def load_sample_trajectory(self):
        """
        Creates a sample trajectory mimicking TUM RGB-D dataset format
        TUM format: timestamp tx ty tz qx qy qz qw
        """
        # Sample starting pose (first frame)
        initial_pose = {
            'timestamp': 1305031102.2754,
            'tx': 1.244, 'ty': 0.831, 'tz': 0.732,
            'qx': 0.013, 'qy': -0.002, 'qz': 0.862, 'qw': 0.506
        }
        
        # Generate a sequence of poses with small motions
        np.random.seed(42)
        n_frames = 50
        trajectory = []
        
        current_pose = initial_pose.copy()
        trajectory.append(current_pose.copy())
        
        for i in range(1, n_frames):
            # Simulate small camera motion between frames
            new_pose = current_pose.copy()
            new_pose['timestamp'] += 0.05  # 20 Hz
            
            # Translation motion (small increments)
            new_pose['tx'] += np.random.uniform(0.02, 0.06)
            new_pose['ty'] += np.random.uniform(-0.02, 0.03)
            new_pose['tz'] += np.random.uniform(-0.01, 0.02)
            
            # Rotation motion (small angle changes)
            # Convert quaternion to rotation matrix, apply small rotation, convert back
            r = Rotation.from_quat([current_pose['qx'], current_pose['qy'], 
                                    current_pose['qz'], current_pose['qw']])
            
            # Small rotation (1-2 degrees)
            delta_rot = Rotation.from_euler('xyz', [0.5, 0.2, 0.3], degrees=True)
            r_new = r * delta_rot
            q_new = r_new.as_quat()  # Returns [x, y, z, w]
            
            new_pose['qx'], new_pose['qy'], new_pose['qz'], new_pose['qw'] = q_new
            
            trajectory.append(new_pose)
            current_pose = new_pose
            
        self.poses = trajectory
        return trajectory
    
    def estimate_motion(self, pose_prev, pose_curr):
        """
        Estimate motion between two poses (translation component)
        Formula: motion = current_position - previous_position
        """
        motion = {
            'dx': pose_curr['tx'] - pose_prev['tx'],
            'dy': pose_curr['ty'] - pose_prev['ty'],
            'dz': pose_curr['tz'] - pose_prev['tz']
        }
        return motion
    
    def predict_next_pose(self, current_pose, motion):
        """
        Predict next pose using current pose and estimated motion
        Formula: new_pose = prev_pose + motion
        """
        predicted_pose = current_pose.copy()
        predicted_pose['tx'] += motion['dx']
        predicted_pose['ty'] += motion['dy']
        predicted_pose['tz'] += motion['dz']
        
        # For simplicity, we assume rotation remains similar
        # In real SLAM, rotation would also be updated
        
        return predicted_pose

# Main execution
print("=" * 70)
print("NVIDIA - Visual SLAM: TUM RGB-D Dataset")
print("=" * 70)

# Initialize SLAM system
slam = TUMRGBDSLAM()

# Load trajectory (simulating TUM dataset sequence)
print("\n📷 TUM RGB-D Dataset Simulation")
print("   Sequence: fr1/xyz (handheld Kinect)")
print("   Format: timestamp tx ty tz qx qy qz qw")
print("   Ground truth from motion capture system")

trajectory = slam.load_sample_trajectory()

print(f"\n📊 First 3 frames of trajectory:")
print("-" * 70)
print(f"{'Frame':<8} {'tx (m)':<10} {'ty (m)':<10} {'tz (m)':<10}")
print("-" * 70)
for i in range(3):
    p = trajectory[i]
    print(f"{i:<8} {p['tx']:<10.3f} {p['ty']:<10.3f} {p['tz']:<10.3f}")

# Visual SLAM: estimate motion between consecutive frames
print("\n🔄 Visual SLAM Motion Estimation:")
print("-" * 70)

# Select two consecutive frames (e.g., frames 10 and 11)
frame_idx = 10
pose_prev = trajectory[frame_idx]
pose_curr = trajectory[frame_idx + 1]

print(f"Frame {frame_idx} (previous):")
print(f"  Position: ({pose_prev['tx']:.3f}, {pose_prev['ty']:.3f}, {pose_prev['tz']:.3f})")
print(f"Frame {frame_idx + 1} (current):")
print(f"  Position: ({pose_curr['tx']:.3f}, {pose_curr['ty']:.3f}, {pose_curr['tz']:.3f})")

# Estimate motion
motion = slam.estimate_motion(pose_prev, pose_curr)
print(f"\n📐 Estimated motion:")
print(f"   dx = {motion['dx']:.3f} m")
print(f"   dy = {motion['dy']:.3f} m")
print(f"   dz = {motion['dz']:.3f} m")

# Predict next pose using formula: new_pose = prev_pose + motion
predicted_pose = slam.predict_next_pose(pose_curr, motion)

print(f"\n🎯 OUTPUT: ({predicted_pose['tx']:.1f}, {predicted_pose['ty']:.1f})")
print(f"   (2D projection from 3D position)")
print(f"   Industry: NVIDIA Visual SLAM")
print(f"\n   Using formula: new_pose = prev_pose + motion")
print(f"   Previous pose: ({pose_curr['tx']:.3f}, {pose_curr['ty']:.3f}, {pose_curr['tz']:.3f})")
print(f"   Motion: ({motion['dx']:.3f}, {motion['dy']:.3f}, {motion['dz']:.3f})")
print(f"   New pose: ({predicted_pose['tx']:.3f}, {predicted_pose['ty']:.3f}, {predicted_pose['tz']:.3f})")

# Validate against actual next frame (if available)
if frame_idx + 2 < len(trajectory):
    actual_next = trajectory[frame_idx + 2]
    print(f"\n✅ Validation with actual next frame:")
    print(f"   Actual:   ({actual_next['tx']:.3f}, {actual_next['ty']:.3f}, {actual_next['tz']:.3f})")
    print(f"   Predicted:({predicted_pose['tx']:.3f}, {predicted_pose['ty']:.3f}, {predicted_pose['tz']:.3f})")
    error = np.sqrt((actual_next['tx']-predicted_pose['tx'])**2 + 
                    (actual_next['ty']-predicted_pose['ty'])**2)
    print(f"   Position error: {error:.3f} m")

# Analyze motion throughout trajectory
print("\n📈 Motion Analysis (all frames):")
print("-" * 50)

motions = []
for i in range(len(trajectory)-1):
    m = slam.estimate_motion(trajectory[i], trajectory[i+1])
    motions.append(m)

avg_motion = {
    'dx': np.mean([m['dx'] for m in motions]),
    'dy': np.mean([m['dy'] for m in motions]),
    'dz': np.mean([m['dz'] for m in motions])
}

print(f"Average motion per frame:")
print(f"  dx = {avg_motion['dx']:.3f} m")
print(f"  dy = {avg_motion['dy']:.3f} m")
print(f"  dz = {avg_motion['dz']:.3f} m")
print(f"  Total distance: {np.sqrt(avg_motion['dx']**2 + avg_motion['dy']**2 + avg_motion['dz']**2):.3f} m/frame")

# Visual SLAM specifics
print("\n🔍 Visual SLAM Pipeline:")
print("   1. Feature extraction (ORB/SIFT) from RGB image")
print("   2. Depth extraction from aligned depth image")
print("   3. Frame-to-frame motion estimation")
print("   4. Trajectory update: new_pose = prev_pose + motion")
print("   5. Loop closure detection and graph optimization")

# Optional: Simple visualization of trajectory
try:
    import matplotlib.pyplot as plt
    
    # Extract trajectory points
    x_vals = [p['tx'] for p in trajectory]
    y_vals = [p['ty'] for p in trajectory]
    z_vals = [p['tz'] for p in trajectory]
    
    fig = plt.figure(figsize=(15, 5))
    
    # 2D trajectory (XY)
    plt.subplot(1, 3, 1)
    plt.plot(x_vals, y_vals, 'b-', linewidth=2)
    plt.plot(x_vals[0], y_vals[0], 'go', markersize=10, label='Start')
    plt.plot(x_vals[-1], y_vals[-1], 'ro', markersize=10, label='End')
    plt.plot(predicted_pose['tx'], predicted_pose['ty'], 'mx', markersize=12, 
             label='Predicted')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Camera Trajectory (XY)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 2D trajectory (XZ)
    plt.subplot(1, 3, 2)
    plt.plot(x_vals, z_vals, 'b-', linewidth=2)
    plt.plot(x_vals[0], z_vals[0], 'go', markersize=10, label='Start')
    plt.plot(x_vals[-1], z_vals[-1], 'ro', markersize=10, label='End')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title('Camera Trajectory (XZ)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 3D trajectory
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.plot(x_vals, y_vals, z_vals, 'b-', linewidth=2)
    ax.plot([x_vals[0]], [y_vals[0]], [z_vals[0]], 'go', markersize=8)
    ax.plot([x_vals[-1]], [y_vals[-1]], [z_vals[-1]], 'ro', markersize=8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Camera Trajectory')
    
    plt.tight_layout()
    plt.savefig('tum_rgbd_slam.png')
    print(f"\n✅ Trajectory visualization saved as 'tum_rgbd_slam.png'")
    
except Exception as e:
    print(f"\n⚠️ Visualization skipped: {e}")

print("\n" + "=" * 70)
print("📌 About TUM RGB-D Dataset:")
print("   • Microsoft Kinect (640x480, 30 Hz)")
print("   • Ground truth from motion capture (8 cameras, 100 Hz)")
print("   • Various indoor sequences: desk, room, plant, etc.")
print("   • Standard benchmark for visual SLAM evaluation")
print("=" * 70)

# How to use real TUM RGB-D data
print("\n🔍 To use real TUM RGB-D data:")
print("   1. Download from: https://vision.in.tum.de/data/datasets/rgbd-dataset/download")
print("   2. Use provided tools: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools")
print("\n   Example Python code for real data:")
print("   ```python")
print("   import numpy as np")
print("   # Load ground truth trajectory")
print("   gt = np.loadtxt('groundtruth.txt')")
print("   # Columns: timestamp tx ty tz qx qy qz qw")
print("   timestamps = gt[:, 0]")
print("   positions = gt[:, 1:4]")
print("   # Load RGB-D frames and run your SLAM system")
print("   ```")
