import numpy as np
import pandas as pd
from pathlib import Path
import zipfile

# Simulated KITTI sensor metadata
# Based on KITTI sensor setup documentation
sensor_data = {
    'sensor': ['Velodyne HDL-64E LiDAR', 'Point Grey Flea2 Video Camera', 'IMU/GPS'],
    'type': ['LiDAR', 'Camera', 'GPS/INS'],
    'max_range_m': [120.0, 50.0, float('inf')],  # LiDAR 120m, Camera effective 50m, GPS unlimited
    'accuracy': [0.02, 0.1, 1.0],  # meters
    'field_of_view': [360, 90, 360],  # degrees
    'data_rate': [10, 15, 100]  # Hz
}

# Create DataFrame
sensors_df = pd.DataFrame(sensor_data)

# Sensor selection logic - compare range
# For autonomous driving, we need max range with practical constraints
# GPS has infinite range but lower accuracy for relative positioning
# Camera range limited by resolution and lighting
# LiDAR offers best balance for obstacle detection

# Filter to practical autonomous driving sensors
practical_sensors = sensors_df[sensors_df['type'].isin(['LiDAR', 'Camera'])]

# Select sensor with maximum range
selected_sensor = practical_sensors.loc[practical_sensors['max_range_m'].idxmax()]

# Output based on selected sensor
if selected_sensor['type'] == 'LiDAR':
    sensor_choice = "LiDAR Selected"
    industry = "Tesla Autopilot"
    reasoning = "Maximum detection range for obstacle detection at highway speeds"
elif selected_sensor['type'] == 'Camera':
    sensor_choice = "Camera Selected" 
    industry = "Tesla Autopilot"
    reasoning = "Selected for visual perception tasks"

# Display results
print("KITTI Sensor Suite Analysis")
print("=" * 50)
print("\nAvailable Sensors:")
print(sensors_df[['sensor', 'type', 'max_range_m']].to_string(index=False))
print("\n" + "=" * 50)
print(f"\nSelected: {sensor_choice}")
print(f"Industry: {industry}")
print(f"Reason: {reasoning}")
print(f"Specs: {selected_sensor['sensor']} - {selected_sensor['max_range_m']}m range, {selected_sensor['accuracy']}m accuracy")
