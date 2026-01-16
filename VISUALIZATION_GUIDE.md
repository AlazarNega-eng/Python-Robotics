# 3D Visualization Guide for KUKA LBR iisy 3 R760

This guide explains all the 3D visualization features available for the robot simulation.

## Quick Start

Run the comprehensive visualization demo:
```bash
python visualize_robot.py
```

This will generate multiple visualization files and open an interactive animation.

## Available Visualization Functions

### 1. Enhanced Robot Plotting

**Function:** `plot_robot(joint_angles, ax=None, show_frames=False, color='blue')`

Enhanced 3D visualization with:
- Gradient-colored links
- Visible joints
- End-effector coordinate frame
- Optional coordinate frames for all joints
- Professional styling

**Example:**
```python
from robotics import KUKALBRiisy3R760
import matplotlib.pyplot as plt

robot = KUKALBRiisy3R760()
joint_angles = np.array([0.3, 0.5, -0.2, 0.4, 0.6, 0.1])

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
robot.plot_robot(joint_angles, ax, show_frames=True)
plt.show()
```

### 2. Multiple Viewing Angles

**Function:** `plot_multiple_views(joint_angles, save_path='robot_multiple_views.png')`

Generates 4 views simultaneously:
- Top view (elevation 90°)
- Front view (elevation 0°, azimuth 0°)
- Side view (elevation 0°, azimuth 90°)
- Isometric view (elevation 30°, azimuth 45°)

**Example:**
```python
robot.plot_multiple_views(joint_angles, 'my_views.png')
```

### 3. Trajectory Visualization

**Function:** `plot_trajectory(trajectory_angles, trajectory_positions=None, ax=None)`

Visualizes robot motion along a trajectory:
- Trajectory path in red
- Start point (green circle)
- End point (red square)
- Robot at key frames

**Example:**
```python
# Generate trajectory
angles, _, _ = robot.generate_trajectory_joint_space(
    start_angles, end_angles, duration=3.0, dt=0.05
)

# Visualize
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
robot.plot_trajectory(angles, ax=ax)
plt.show()
```

### 4. Workspace Visualization

**Function:** `plot_workspace(n_samples=1000, ax=None)`

Visualizes the robot's reachable workspace by sampling random configurations:
- Color-coded by height (Z-axis)
- Shows workspace density
- Base position marked

**Example:**
```python
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
robot.plot_workspace(n_samples=2000, ax=ax)
plt.show()
```

### 5. Trajectory Animation

**Function:** `animate_trajectory(trajectory_angles, interval=50, save_gif=False, filename='robot_animation.gif')`

Creates an animated visualization of robot motion:
- Smooth animation of trajectory
- Optional GIF export
- Interactive playback

**Example:**
```python
# Generate trajectory
angles, _, _ = robot.generate_trajectory_joint_space(
    start_angles, end_angles, duration=5.0, dt=0.1
)

# Animate
anim = robot.animate_trajectory(angles, interval=100, save_gif=True)
plt.show()
```

## Generated Visualization Files

When you run `visualize_robot.py`, the following files are created:

1. **robot_3d_enhanced.png** - Single enhanced 3D view with coordinate frames
2. **robot_multiple_views.png** - Four different viewing angles
3. **robot_trajectory.png** - Trajectory path with robot at key frames
4. **robot_workspace.png** - Reachable workspace visualization
5. **robot_configurations.png** - Comparison of multiple robot configurations
6. **robot_ee_path.png** - End-effector path with time coloring

## Customization Options

### Viewing Angles

You can manually set viewing angles:
```python
ax.view_init(elev=30, azim=45)  # elevation and azimuth in degrees
```

### Color Schemes

The robot uses gradient colors by default. You can customize:
- Link colors: Modify the `colors_list` in `plot_robot()`
- End-effector: Red star marker
- Base: Dark gray sphere

### Coordinate Frames

Enable coordinate frames for all joints:
```python
robot.plot_robot(joint_angles, ax, show_frames=True)
```

This shows:
- Red: X-axis
- Green: Y-axis
- Blue: Z-axis

## Tips for Best Results

1. **High Resolution**: Use `dpi=150` or higher for publication-quality images
2. **Aspect Ratio**: The code automatically sets equal aspect ratios
3. **Limits**: Workspace limits are set to ±0.8m (maximum reach is 0.76m)
4. **Animation**: For smooth animations, use `dt=0.05-0.1` in trajectory generation
5. **Workspace**: Use 1000-2000 samples for good workspace visualization

## Interactive Features

When using `plt.show()`, you can:
- Rotate: Click and drag
- Zoom: Scroll wheel
- Pan: Right-click and drag
- Reset: Use toolbar buttons

## Advanced Usage

### Custom Trajectory Visualization

```python
# Compute end-effector positions
ee_positions = []
for angles in trajectory_angles:
    pos, _ = robot.forward_kinematics(angles)
    ee_positions.append(pos)
ee_positions = np.array(ee_positions)

# Custom plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 'r-', linewidth=2)
robot.plot_robot(trajectory_angles[0], ax)  # Robot at start
robot.plot_robot(trajectory_angles[-1], ax)  # Robot at end
plt.show()
```

### Multiple Robots in One Plot

```python
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot multiple configurations
robot.plot_robot(config1, ax)
robot.plot_robot(config2, ax)
robot.plot_robot(config3, ax)

plt.show()
```

## Troubleshooting

**Issue:** Animation window doesn't appear
- **Solution:** Make sure you have a display/GUI available. On headless systems, use `save_gif=True`

**Issue:** GIF saving fails
- **Solution:** Install Pillow: `pip install pillow`

**Issue:** Poor quality images
- **Solution:** Increase DPI: `plt.savefig('file.png', dpi=300)`

**Issue:** Slow rendering
- **Solution:** Reduce number of samples in workspace visualization or trajectory points

## Examples in Code

See `visualize_robot.py` for comprehensive examples of all visualization features.
