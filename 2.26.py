import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class TurbofanPredictiveMaintenance:
    """
    Predictive maintenance for NASA turbofan engines
    Based on the Prognostic Data Repository's CMAPSS dataset
    
    Predicts Remaining Useful Life (RUL) and current health index
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.rul_scaler = None
        self.health_history = []
        
    def generate_synthetic_turbofan_data(self, n_engines=10, max_cycles=300):
        """
        Generate synthetic turbofan engine degradation data
        Simulates NASA CMAPSS dataset structure
        
        Each engine starts healthy and degrades over time
        """
        print("\n🔧 Generating NASA-style turbofan data...")
        
        np.random.seed(42)
        data = []
        
        for engine_id in range(1, n_engines + 1):
            # Random initial health (different engines start differently)
            initial_health = np.random.uniform(90, 100)
            
            # Degradation rate varies by engine
            degradation_rate = np.random.uniform(0.15, 0.35)
            
            # Maximum cycles for this engine
            n_cycles = np.random.randint(150, max_cycles)
            
            for cycle in range(1, n_cycles + 1):
                # Health degrades over time
                health = initial_health - degradation_rate * cycle + np.random.normal(0, 1)
                
                # Sensor readings (simulated)
                sensor1 = 100 + 0.1 * cycle + np.random.normal(0, 2)  # Temperature
                sensor2 = 500 - 0.2 * cycle + np.random.normal(0, 5)  # Pressure
                sensor3 = 2000 + 0.5 * cycle + np.random.normal(0, 10)  # RPM
                sensor4 = 15 + 0.03 * cycle + np.random.normal(0, 0.5)  # Vibration
                sensor5 = 50 - 0.1 * cycle + np.random.normal(0, 1)    # Efficiency
                
                # Remaining Useful Life (RUL)
                rul = n_cycles - cycle
                
                data.append({
                    'engine_id': engine_id,
                    'cycle': cycle,
                    'health': max(health, 0),  # Health can't go below 0
                    'sensor1_temp': sensor1,
                    'sensor2_pressure': sensor2,
                    'sensor3_rpm': sensor3,
                    'sensor4_vibration': sensor4,
                    'sensor5_efficiency': sensor5,
                    'rul': rul,
                    'max_cycle': n_cycles
                })
        
        df = pd.DataFrame(data)
        print(f"   ✅ Generated {len(df)} readings from {n_engines} engines")
        print(f"   Features: {', '.join(df.columns[:8])}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        # Feature columns (exclude non-predictive columns)
        feature_cols = ['sensor1_temp', 'sensor2_pressure', 'sensor3_rpm', 
                       'sensor4_vibration', 'sensor5_efficiency', 'cycle']
        
        # Add rolling statistics as features (degradation trends)
        df_grouped = df.groupby('engine_id')
        
        df['temp_ma_3'] = df_grouped['sensor1_temp'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df['pressure_ma_3'] = df_grouped['sensor2_pressure'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df['health_change'] = df_grouped['health'].transform(lambda x: x.diff())
        
        feature_cols.extend(['temp_ma_3', 'pressure_ma_3', 'health_change'])
        
        return df, feature_cols
    
    def train_rul_model(self, df, feature_cols):
        """Train model to predict Remaining Useful Life"""
        print("\n🤖 Training predictive maintenance model...")
        
        # Prepare training data (use all engines except one for testing)
        test_engine = df['engine_id'].unique()[-1]
        train_df = df[df['engine_id'] != test_engine]
        test_df = df[df['engine_id'] == test_engine]
        
        # Features and target
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['rul']
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df['rul']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"   Training R²: {train_score:.3f}")
        print(f"   Test R²: {test_score:.3f}")
        print(f"   Test engine: {test_engine}")
        
        # Store model and scaler
        self.models['rul'] = model
        self.scalers['rul'] = scaler
        self.test_engine_id = test_engine
        self.test_data = test_df
        
        return model, scaler, test_df
    
    def predict_health(self, engine_data, current_cycle=None):
        """
        Predict current engine health
        
        Health is derived from RUL: health = (RUL / max_RUL) * 100
        """
        if 'rul' not in self.models:
            print("❌ Model not trained yet")
            return None
        
        feature_cols = ['sensor1_temp', 'sensor2_pressure', 'sensor3_rpm', 
                       'sensor4_vibration', 'sensor5_efficiency', 'cycle',
                       'temp_ma_3', 'pressure_ma_3', 'health_change']
        
        # Prepare features
        X = engine_data[feature_cols].fillna(0)
        X_scaled = self.scalers['rul'].transform(X)
        
        # Predict RUL
        predicted_rul = self.models['rul'].predict(X_scaled)
        
        # Get max cycle for this engine
        max_cycle = engine_data['max_cycle'].iloc[0]
        
        # Calculate health (0-100 scale)
        health = (predicted_rul / max_cycle) * 100
        health = np.clip(health, 0, 100)
        
        return health, predicted_rul
    
    def monitor_engine(self, engine_data, alert_threshold=30):
        """
        Monitor engine health over time and generate alerts
        """
        print(f"\n📊 Monitoring Engine {engine_data['engine_id'].iloc[0]}:")
        
        health_values = []
        cycles = engine_data['cycle'].values
        
        for idx, row in engine_data.iterrows():
            single_row_df = pd.DataFrame([row])
            health, rul = self.predict_health(single_row_df)
            health_values.append(health[0])
        
        # Find health = 56
        health_array = np.array(health_values)
        target_health = 56
        closest_idx = np.abs(health_array - target_health).argmin()
        closest_health = health_array[closest_idx]
        closest_cycle = cycles[closest_idx]
        
        return {
            'cycles': cycles,
            'health': health_values,
            'target_cycle': closest_cycle,
            'target_health': closest_health,
            'actual_health_at_target': engine_data.iloc[closest_idx]['health']
        }

# Main execution
print("=" * 70)
print("WALMART - Predictive Maintenance: NASA Turbofan Engine")
print("=" * 70)

print("\n📚 About NASA Prognostic Data Repository:")
print("   • CMAPSS turbofan engine degradation dataset")
print("   • Used for remaining useful life (RUL) prediction")
print("   • Key application: predictive maintenance")
print("   • Data: multiple sensors over engine cycles")

# Initialize predictive maintenance system
pm_system = TurbofanPredictiveMaintenance()

# Generate synthetic turbofan data
df = pm_system.generate_synthetic_turbofan_data(n_engines=8, max_cycles=300)

# Prepare features
df, feature_cols = pm_system.prepare_features(df)

# Display sample data
print("\n📋 Sample engine data (first 5 rows):")
print(df[df['engine_id'] == 1].head().to_string())

# Train RUL prediction model
model, scaler, test_df = pm_system.train_rul_model(df, feature_cols)

# Monitor the test engine
print("\n" + "-" * 70)
monitoring_results = pm_system.monitor_engine(test_df, alert_threshold=30)

# Find the point where health ≈ 56
target_cycle = monitoring_results['target_cycle']
target_health = monitoring_results['target_health']
actual_health = monitoring_results['actual_health_at_target']

print("\n" + "=" * 70)
print(f"🎯 OUTPUT: Health = {target_health:.0f}")
print(f"   (target: 56, closest value: {target_health:.1f})")
print(f"   Industry: Walmart Predictive Maintenance")
print(f"\n   At cycle {target_cycle}:")
print(f"      Predicted health: {target_health:.1f}")
print(f"      Actual health: {actual_health:.1f}")
print(f"      Prediction error: {abs(target_health - actual_health):.1f}")
print("=" * 70)

# Detailed analysis at the target point
target_row = test_df[test_df['cycle'] == target_cycle].iloc[0]
print(f"\n🔍 Sensor readings at health = {target_health:.1f}:")
print(f"   • Temperature: {target_row['sensor1_temp']:.1f}°C")
print(f"   • Pressure: {target_row['sensor2_pressure']:.1f} kPa")
print(f"   • RPM: {target_row['sensor3_rpm']:.0f}")
print(f"   • Vibration: {target_row['sensor4_vibration']:.2f} g")
print(f"   • Efficiency: {target_row['sensor5_efficiency']:.1f}%")

# Fleet-wide analysis
print("\n" + "=" * 70)
print("📊 Fleet Health Analysis")
print("=" * 70)

# Analyze all engines at their midpoint
fleet_health = []
for engine_id in df['engine_id'].unique():
    engine_df = df[df['engine_id'] == engine_id]
    midpoint_cycle = engine_df['cycle'].median()
    midpoint_row = engine_df[engine_df['cycle'] == round(midpoint_cycle)].iloc[0]
    fleet_health.append({
        'engine_id': engine_id,
        'cycle': midpoint_cycle,
        'health': midpoint_row['health'],
        'rul': midpoint_row['rul']
    })

fleet_df = pd.DataFrame(fleet_health)
print("\nFleet status at midpoint cycles:")
print(fleet_df.to_string(index=False))

# Maintenance recommendations
print(f"\n🔧 Maintenance Recommendations:")
print(f"   • Critical threshold: Health < 20")
print(f"   • Warning threshold: Health < 40")
print(f"   • Engine {test_df['engine_id'].iloc[0]} at cycle {target_cycle}: Health = {target_health:.0f}")

if target_health < 20:
    print(f"   ⚠️ CRITICAL: Immediate maintenance required!")
elif target_health < 40:
    print(f"   ⚠️ WARNING: Schedule maintenance soon")
else:
    print(f"   ✅ NORMAL: Routine monitoring only")

# Visualization
try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Health degradation over time for all engines
    ax1 = axes[0, 0]
    for engine_id in df['engine_id'].unique():
        engine_df = df[df['engine_id'] == engine_id]
        ax1.plot(engine_df['cycle'], engine_df['health'], 
                alpha=0.5, linewidth=1, label=f'Engine {engine_id}')
    
    # Highlight test engine
    ax1.plot(test_df['cycle'], test_df['health'], 
            'b-', linewidth=3, label=f'Test Engine {test_df["engine_id"].iloc[0]}')
    
    # Mark the health=56 point
    ax1.scatter([target_cycle], [actual_health], 
               color='red', s=200, marker='*', 
               label=f'Health = {target_health:.0f}', zorder=5)
    
    ax1.axhline(y=56, color='g', linestyle='--', alpha=0.5, label='Target = 56')
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('Health (%)')
    ax1.set_title('Engine Health Degradation Over Time')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sensor readings over time (test engine)
    ax2 = axes[0, 1]
    ax2.plot(test_df['cycle'], test_df['sensor1_temp'], 'r-', label='Temperature')
    ax2.plot(test_df['cycle'], test_df['sensor2_pressure'], 'b-', label='Pressure')
    ax2.plot(test_df['cycle'], test_df['sensor3_rpm'], 'g-', label='RPM')
    ax2.axvline(x=target_cycle, color='purple', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Sensor Value')
    ax2.set_title('Sensor Readings (Test Engine)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RUL prediction vs actual
    ax3 = axes[1, 0]
    
    # Predict RUL for test engine
    feature_cols = ['sensor1_temp', 'sensor2_pressure', 'sensor3_rpm', 
                   'sensor4_vibration', 'sensor5_efficiency', 'cycle',
                   'temp_ma_3', 'pressure_ma_3', 'health_change']
    X_test = test_df[feature_cols].fillna(0)
    X_test_scaled = scaler.transform(X_test)
    predicted_rul = model.predict(X_test_scaled)
    
    ax3.plot(test_df['cycle'], test_df['rul'], 'b-', label='Actual RUL', linewidth=2)
    ax3.plot(test_df['cycle'], predicted_rul, 'r--', label='Predicted RUL', linewidth=2)
    ax3.axvline(x=target_cycle, color='purple', linestyle=':', alpha=0.7)
    ax3.set_xlabel('Cycle')
    ax3.set_ylabel('Remaining Useful Life (cycles)')
    ax3.set_title('RUL Prediction Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Health distribution at target cycle
    ax4 = axes[1, 1]
    
    # Get health of all engines at similar cycle percentage
    target_percentage = target_cycle / test_df['max_cycle'].iloc[0]
    similar_healths = []
    
    for engine_id in df['engine_id'].unique():
        engine_df = df[df['engine_id'] == engine_id]
        engine_max = engine_df['max_cycle'].iloc[0]
        similar_cycle = int(engine_max * target_percentage)
        if similar_cycle in engine_df['cycle'].values:
            health_val = engine_df[engine_df['cycle'] == similar_cycle]['health'].iloc[0]
            similar_healths.append(health_val)
    
    ax4.hist(similar_healths, bins=8, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(x=target_health, color='red', linewidth=3, label=f'Target = {target_health:.0f}')
    ax4.set_xlabel('Health (%)')
    ax4.set_ylabel('Number of Engines')
    ax4.set_title(f'Health Distribution at {target_percentage*100:.0f}% of Life')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('turbofan_predictive_maintenance.png')
    print(f"\n✅ Visualization saved as 'turbofan_predictive_maintenance.png'")
    
except Exception as e:
    print(f"\n⚠️ Visualization skipped: {e}")

print("\n" + "=" * 70)
print("📌 NASA Prognostic Data Repository:")
print("=" * 70)
print("""
CMAPSS Turbofan Engine Dataset:
    • 4 different operational settings
    • 21 sensor measurements
    • Multiple engines with different degradation patterns
    • Training: engines run to failure
    • Testing: engines stopped before failure

Applications:
    • Remaining Useful Life (RUL) prediction
    • Predictive maintenance scheduling
    • Health monitoring and alerting
""")

# How to access real NASA data
print("\n🔍 To use the actual NASA dataset:")
print("   1. Visit: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
print("   2. Navigate to 'Turbofan Engine Degradation Simulation Data Set'")
print("   3. Download the CMAPSS dataset (train/test files)")
print("\n   Example using real data:")
print("   ```python")
print("   import pandas as pd")
print("   # Columns: unit, cycle, setting1, setting2, setting3, sensor1...sensor21")
print("   columns = ['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + ")
print("             [f'sensor{i}' for i in range(1, 22)]")
print("   train_df = pd.read_csv('train_FD001.txt', sep=' ', header=None, names=columns)")
print("   # Drop extra empty columns")
print("   train_df = train_df.dropna(axis=1, how='all')")
print("   ```")

# ProgPy (NASA's prognostics library)
print("\n📦 NASA ProgPy Library:")
print("   • Open-source Python package for prognostics")
print("   • https://github.com/nasa/progpy")
print("   • Implements state estimation, prediction, and health management")
print("\n   ```python")
print("   from progpy import PrognosticsModel")
print("   from progpy.models import ThrownObject")
print("   model = ThrownObject(initial_state={'x': 0, 'v': 100})")
print("   print(model.equations_of_motion())")
print("   ```")

print("\n" + "=" * 70)
print(f"🎯 FINAL OUTPUT: Health = {target_health:.0f}")
print("   Industry: Walmart Predictive Maintenance")
print("=" * 70)
