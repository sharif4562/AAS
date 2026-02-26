import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

@dataclass
class PIDGains:
    """PID controller gains"""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain

class PIDController:
    """
    PID (Proportional-Integral-Derivative) Controller
    Based on Arduino-PID-Library principles
    
    A feedback control system that continuously computes an error value
    as the difference between desired setpoint and measured process variable
    """
    
    def __init__(self, gains: PIDGains, setpoint: float = 0.0, 
                 sample_time: float = 0.1, output_limits=(-100, 100)):
        """
        Initialize PID controller
        
        Args:
            gains: PIDGains object with kp, ki, kd values
            setpoint: desired target value
            sample_time: time between controller updates (seconds)
            output_limits: min/max output limits
        """
        self.gains = gains
        self.setpoint = setpoint
        self.sample_time = sample_time
        self.output_limits = output_limits
        
        # Controller state
        self.last_time = time.time()
        self.last_error = 0.0
        self.integral = 0.0
        self.last_output = 0.0
        
        # History for analysis
        self.history = {
            'time': [],
            'setpoint': [],
            'measurement': [],
            'error': [],
            'output': [],
            'p_term': [],
            'i_term': [],
            'd_term': []
        }
        
        print(f"🔧 PID Controller Initialized:")
        print(f"   • Gains: Kp={gains.kp}, Ki={gains.ki}, Kd={gains.kd}")
        print(f"   • Setpoint: {setpoint}")
        print(f"   • Sample time: {sample_time}s")
        
    def compute(self, measurement: float, current_time: float = None) -> float:
        """
        Compute PID output based on measurement
        
        Args:
            measurement: current process variable measurement
            current_time: current timestamp (optional)
            
        Returns:
            controller output
        """
        if current_time is None:
            current_time = time.time()
            
        # Time difference since last computation
        dt = current_time - self.last_time
        
        # Only compute if sample time has elapsed
        if dt < self.sample_time and self.last_time > 0:
            return self.last_output
        
        # Calculate error
        error = self.setpoint - measurement
        
        # Proportional term
        p_term = self.gains.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        i_term = self.gains.ki * self.integral
        
        # Derivative term (using derivative on measurement to avoid derivative kick)
        d_term = self.gains.kd * (self.last_error - error) / dt if dt > 0 else 0
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits
        output = max(min(output, self.output_limits[1]), self.output_limits[0])
        
        # Anti-windup: clamp integral if output is saturated
        if output != p_term + i_term + d_term:
            self.integral -= error * dt  # Undo integration
        
        # Update state
        self.last_time = current_time
        self.last_error = error
        self.last_output = output
        
        # Store history
        self.history['time'].append(current_time)
        self.history['setpoint'].append(self.setpoint)
        self.history['measurement'].append(measurement)
        self.history['error'].append(error)
        self.history['output'].append(output)
        self.history['p_term'].append(p_term)
        self.history['i_term'].append(i_term)
        self.history['d_term'].append(d_term)
        
        return output
    
    def set_setpoint(self, setpoint: float):
        """Change the setpoint"""
        self.setpoint = setpoint
        print(f"🎯 Setpoint changed to: {setpoint}")
        
    def reset(self):
        """Reset controller state"""
        self.last_time = time.time()
        self.last_error = 0.0
        self.integral = 0.0
        self.last_output = 0.0
        print("🔄 PID Controller reset")
        
    def get_error(self) -> float:
        """Get current error (difference between setpoint and last measurement)"""
        if len(self.history['error']) > 0:
            return self.history['error'][-1]
        return 0.0

class ThermalSystem:
    """
    Simulated thermal system (first-order with time delay)
    Used as the process to control
    """
    
    def __init__(self, initial_temp: float = 20.0, ambient_temp: float = 25.0,
                 time_constant: float = 5.0, noise_amplitude: float = 0.5):
        """
        Initialize thermal system
        
        Args:
            initial_temp: starting temperature (°C)
            ambient_temp: ambient temperature (°C)
            time_constant: system time constant (seconds)
            noise_amplitude: measurement noise amplitude
        """
        self.temperature = initial_temp
        self.ambient_temp = ambient_temp
        self.time_constant = time_constant
        self.noise_amplitude = noise_amplitude
        self.heating_power = 0.0
        
    def update(self, control_signal: float, dt: float) -> float:
        """
        Update system state based on control signal
        
        Args:
            control_signal: heating/cooling power (-100 to 100)
            dt: time step (seconds)
            
        Returns:
            measured temperature with noise
        """
        # Apply control signal (clamp to valid range)
        self.heating_power = max(min(control_signal, 100), -100)
        
        # First-order thermal dynamics
        dT_dt = (self.heating_power * 0.1 + self.ambient_temp - self.temperature) / self.time_constant
        self.temperature += dT_dt * dt
        
        # Add measurement noise
        measurement = self.temperature + np.random.normal(0, self.noise_amplitude)
        
        return measurement

# Main execution
print("=" * 70)
print("MICROSOFT - Feedback Control: PID Control Dataset")
print("=" * 70)

print("\n📚 About PID Control:")
print("   • Proportional-Integral-Derivative controller")
print("   • Most common feedback control algorithm")
print("   • Used in: industrial automation, robotics, HVAC, etc.")
print("   • Error = Setpoint - Measurement")

# Create PID controller
pid = PIDController(
    gains=PIDGains(kp=2.0, ki=0.5, kd=0.1),
    setpoint=30.0,  # Target temperature 30°C
    sample_time=0.1,
    output_limits=(-100, 100)
)

# Create thermal system to control
system = ThermalSystem(
    initial_temp=20.0,
    ambient_temp=25.0,
    time_constant=3.0,
    noise_amplitude=0.3
)

print("\n" + "-" * 70)
print("📊 Simulation: Temperature Control")
print("-" * 70)

# Run simulation
simulation_time = 30  # seconds
dt = 0.1
steps = int(simulation_time / dt)

print(f"\n⏱️  Running {simulation_time}s simulation...")
print(f"   Target: {pid.setpoint}°C")
print(f"   Initial temperature: {system.temperature:.1f}°C")

temperatures = []
errors = []
outputs = []
times = []

for i in range(steps):
    current_time = i * dt
    
    # Measure current temperature
    measurement = system.temperature
    
    # Compute control signal
    control = pid.compute(measurement, current_time)
    
    # Apply control to system
    system.update(control, dt)
    
    # Store data
    temperatures.append(measurement)
    errors.append(pid.get_error())
    outputs.append(control)
    times.append(current_time)
    
    # Print progress at intervals
    if i % 100 == 0:
        print(f"   t={current_time:.1f}s: Temp={measurement:.2f}°C, "
              f"Error={pid.get_error():.2f}°C, Output={control:.1f}")

# Find error at specific time for output
target_time = 15.0  # 15 seconds into simulation
target_idx = int(target_time / dt)
error_at_target = errors[target_idx]

print("\n" + "-" * 50)
print(f"🎯 OUTPUT: Error = {error_at_target:.0f}")
print(f"   Industry: Microsoft Azure IoT")
print(f"\n   At t={target_time:.1f}s:")
print(f"      Setpoint: {pid.setpoint:.1f}°C")
print(f"      Measurement: {temperatures[target_idx]:.2f}°C")
print(f"      Error: {error_at_target:.2f}°C")
print(f"      Control output: {outputs[target_idx]:.1f}")

# Analysis of control performance
print("\n" + "=" * 70)
print("📈 Control Performance Analysis")
print("=" * 70)

# Steady-state error
steady_state_start = int(20 / dt)  # Last 10 seconds
steady_state_errors = errors[steady_state_start:]
avg_error = np.mean(np.abs(steady_state_errors))
max_error = np.max(np.abs(errors))
settling_time = None

# Find settling time (when error stays within 5% of setpoint)
threshold = 0.05 * pid.setpoint
for i, error in enumerate(errors):
    if np.abs(error) < threshold:
        # Check if it stays within threshold
        if all(np.abs(errors[j]) < threshold for j in range(i, min(i+50, len(errors)))):
            settling_time = times[i]
            break

print(f"\n📊 Metrics:")
print(f"   • Maximum absolute error: {max_error:.2f}°C")
print(f"   • Steady-state error: {avg_error:.2f}°C")
print(f"   • Settling time: {settling_time:.1f}s" if settling_time else "   • Settling time: Not reached")
print(f"   • Final error: {errors[-1]:.2f}°C")

# PID term analysis
print(f"\n🔍 PID Term Analysis (at peak):")
max_error_idx = np.argmax(np.abs(errors))
print(f"   At t={times[max_error_idx]:.1f}s:")
print(f"      Error = {errors[max_error_idx]:.2f}°C")
print(f"      P term = {pid.history['p_term'][max_error_idx]:.2f}")
print(f"      I term = {pid.history['i_term'][max_error_idx]:.2f}")
print(f"      D term = {pid.history['d_term'][max_error_idx]:.2f}")
print(f"      Total = {outputs[max_error_idx]:.2f}")

# Visualize
try:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Temperature tracking
    ax1 = axes[0]
    ax1.plot(times, temperatures, 'b-', linewidth=2, label='Measured Temperature')
    ax1.axhline(y=pid.setpoint, color='r', linestyle='--', label=f'Setpoint ({pid.setpoint}°C)')
    ax1.axvline(x=target_time, color='g', linestyle=':', alpha=0.7, label=f't={target_time}s')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('PID Temperature Control')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error over time
    ax2 = axes[1]
    ax2.plot(times, errors, 'r-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=target_time, color='g', linestyle=':', alpha=0.7)
    ax2.fill_between(times, 0, errors, where=np.array(errors) > 0, 
                     color='red', alpha=0.3, label='Positive error')
    ax2.fill_between(times, 0, errors, where=np.array(errors) < 0, 
                     color='blue', alpha=0.3, label='Negative error')
    ax2.scatter([target_time], [error_at_target], color='purple', 
                s=100, zorder=5, label=f'Error = {error_at_target:.1f}')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (°C)')
    ax2.set_title('Control Error (Setpoint - Measurement)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: PID components
    ax3 = axes[2]
    ax3.plot(times, pid.history['p_term'], 'r-', alpha=0.7, label='Proportional')
    ax3.plot(times, pid.history['i_term'], 'b-', alpha=0.7, label='Integral')
    ax3.plot(times, pid.history['d_term'], 'g-', alpha=0.7, label='Derivative')
    ax3.plot(times, outputs, 'k--', linewidth=2, label='Total Output')
    ax3.axvline(x=target_time, color='g', linestyle=':', alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control Signal')
    ax3.set_title('PID Components')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pid_control.png')
    print(f"\n✅ Visualization saved as 'pid_control.png'")
    
except Exception as e:
    print(f"\n⚠️ Visualization skipped: {e}")

print("\n" + "=" * 70)
print("📌 PID Control Theory:")
print("=" * 70)
print("""
PID Controller Equation:
    u(t) = Kp·e(t) + Ki·∫e(τ)dτ + Kd·de/dt
    
Where:
    u(t) = control output
    e(t) = error = setpoint - measurement
    Kp = proportional gain
    Ki = integral gain
    Kd = derivative gain

The error is the fundamental quantity in feedback control:
    • Positive error: measurement below setpoint
    • Negative error: measurement above setpoint
    • Zero error: perfect tracking
""")

# How to use with Arduino-PID-Library
print("\n🔍 Using with Arduino-PID-Library:")
print("=" * 70)
print("\n📥 Install Arduino PID Library:")
print("```bash")
print("# Using Arduino Library Manager")
print("Sketch → Include Library → Manage Libraries → Search 'PID' → Install")
print("```")

print("\n📝 Arduino code example:")
print("```cpp")
print("#include <PID_v1.h>")
print("")
print("// PID variables")
print("double setpoint = 30.0;  // Target temperature")
print("double measurement;       // Current temperature")
print("double output;           // Control signal")
print("")
print("// PID gains")
print("double Kp = 2.0, Ki = 0.5, Kd = 0.1;")
print("")
print("// Create PID controller")
print("PID myPID(&measurement, &output, &setpoint, Kp, Ki, Kd, DIRECT);")
print("")
print("void setup() {")
print("  myPID.SetMode(AUTOMATIC);")
print("  myPID.SetOutputLimits(-100, 100);")
print("  myPID.SetSampleTime(100);  // 100ms sample time")
print("}")
print("")
print("void loop() {")
print("  measurement = readTemperature();  // Read sensor")
print("  myPID.Compute();                  // Compute PID output")
print("  applyControl(output);             // Apply to actuator")
print("  ")
print("  // The error is automatically handled by PID")
print("  // Error = setpoint - measurement")
print("  ")
print("  Serial.print(\"Error: \");")
print("  Serial.println(setpoint - measurement);")
print("  delay(100);")
print("}")
print("```")

print("\n📊 PID Tuning Guide:")
print("   • Increase Kp: faster response, more overshoot")
print("   • Increase Ki: eliminates steady-state error, more oscillation")
print("   • Increase Kd: damping, noise amplification")
print("   • Common tuning methods: Ziegler-Nichols, Cohen-Coon, trial and error")

print("\n" + "=" * 70)
print("🎯 SUMMARY")
print("=" * 70)
print(f"✓ PID Controller implemented with Kp={pid.gains.kp}, Ki={pid.gains.ki}, Kd={pid.gains.kd}")
print(f"✓ Simulated thermal system with time constant {system.time_constant}s")
print(f"✓ Error at t={target_time}s: {error_at_target:.2f}°C")
print(f"✓ Output: Error = {error_at_target:.0f}")
print("✓ Industry: Microsoft Azure IoT")
print("=" * 70)
