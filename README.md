# KUKA LBR iisy 3 R760 Robot Simulation

Comprehensive Python simulation and implementation for the KUKA LBR iisy 3 R760 6 DOF collaborative robot.

## Features

This implementation includes all core robotics components:

1. **Forward Kinematics** - Compute end-effector pose from joint angles using Modified Denavit-Hartenberg (MDH) parameters
2. **Inverse Kinematics** - Numerical and geometric methods to compute joint angles from desired end-effector pose
3. **Jacobian** - Geometric Jacobian computation for velocity and force mapping
4. **Dynamics** - Mass matrix, Coriolis/centrifugal forces, and gravity compensation
5. **Trajectory Generation** - Linear, cubic, and quintic polynomial trajectories in joint and Cartesian space
6. **Control** - Low-level PID control and high-level computed torque/operational space control

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from robotics import KUKALBRiisy3R760
import numpy as np

# Initialize robot
robot = KUKALBRiisy3R760()

# Forward Kinematics
joint_angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
position, orientation = robot.forward_kinematics(joint_angles)
print(f"End-effector position: {position}")

# Inverse Kinematics
target_pos = np.array([0.3, 0.2, 0.5])
target_orient = np.eye(3)
ik_solution, success = robot.inverse_kinematics_numerical(
    target_pos, target_orient
)
print(f"IK solution: {ik_solution}")

# Dynamics
joint_velocities = np.zeros(6)
joint_accelerations = np.zeros(6)
torques = robot.compute_dynamics(
    joint_angles, joint_velocities, joint_accelerations
)
print(f"Required torques: {torques}")
```

### Run Example

```bash
python robotics.py
```

This will run a comprehensive example demonstrating all features and generate a visualization.

## Robot Specifications

- **Maximum reach:** 760 mm
- **Rated payload:** 3 kg
- **Number of axes:** 6
- **Pose repeatability:** ±0.1 mm
- **Weight:** 22.8 kg

## MDH Parameters

The robot uses Modified Denavit-Hartenberg parameters:

| Joint | α (deg) | a (m) | d (m) |
|-------|---------|-------|-------|
| 1     | +90     | 0.000 | 0.340 |
| 2     | 0       | 0.000 | 0.000 |
| 3     | 0       | 0.000 | 0.000 |
| 4     | +90     | 0.000 | 0.300 |
| 5     | -90     | 0.000 | 0.000 |
| 6     | 0       | 0.000 | 0.120 |

## API Reference

### Forward Kinematics

- `forward_kinematics(joint_angles)`: Compute end-effector pose
- `forward_kinematics_all_joints(joint_angles)`: Get all intermediate frames

### Inverse Kinematics

- `inverse_kinematics_numerical(target_pos, target_orient, ...)`: Numerical IK using Jacobian
- `inverse_kinematics_geometric(target_pos, target_orient, ...)`: Geometric IK using optimization

### Jacobian

- `jacobian(joint_angles)`: Compute 6x6 geometric Jacobian
- `jacobian_dot(joint_angles, joint_velocities)`: Compute Jacobian time derivative

### Dynamics

- `compute_mass_matrix(joint_angles)`: Compute 6x6 mass matrix M(q)
- `compute_coriolis_matrix(joint_angles, joint_velocities)`: Compute Coriolis matrix C(q, q_dot)
- `compute_gravity_vector(joint_angles)`: Compute gravity vector G(q)
- `compute_dynamics(...)`: Compute joint torques using Euler-Lagrange equations

### Trajectory Generation

- `generate_trajectory_linear(start_pos, end_pos, duration)`: Linear Cartesian trajectory
- `generate_trajectory_joint_space(start_angles, end_angles, duration)`: Cubic joint space trajectory
- `generate_trajectory_quintic(...)`: Quintic polynomial trajectory with velocity constraints

### Control

- `PIDController`: Low-level PID controller class
- `compute_control_torque_pid(...)`: PID-based joint control
- `compute_control_torque_computed_torque(...)`: Computed torque control
- `compute_control_torque_operational_space(...)`: Operational space control

### Visualization

- `plot_robot(joint_angles)`: 3D visualization of robot configuration

## Notes

- Joint angles are in radians
- Positions are in meters
- Torques are in N⋅m
- The dynamics model uses simplified inertia approximations - adjust based on actual robot specifications
- Joint limits are enforced in IK and control algorithms

## License

This code is provided for educational and research purposes.
