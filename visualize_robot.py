"""
3D Visualization Demo for KUKA LBR iisy 3 R760 Robot
====================================================

This script demonstrates various 3D visualization capabilities:
- Static robot configurations
- Multiple viewing angles
- Trajectory visualization
- Workspace visualization
- Animation (optional)
"""

import numpy as np
import matplotlib.pyplot as plt
from robotics import KUKALBRiisy3R760

def main():
    """Main visualization demo"""
    
    print("=" * 60)
    print("KUKA LBR iisy 3 R760 - 3D Visualization Demo")
    print("=" * 60)
    
    # Initialize robot
    robot = KUKALBRiisy3R760()
    
    # ========================================================================
    # 1. Static Robot Configuration with Enhanced Visualization
    # ========================================================================
    print("\n1. Generating enhanced robot visualization...")
    joint_angles = np.array([0.3, 0.5, -0.2, 0.4, 0.6, 0.1])
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    robot.plot_robot(joint_angles, ax, show_frames=True)
    plt.savefig('robot_3d_enhanced.png', dpi=150, bbox_inches='tight')
    print("   Saved: robot_3d_enhanced.png")
    plt.close()
    
    # ========================================================================
    # 2. Multiple Viewing Angles
    # ========================================================================
    print("\n2. Generating multiple viewing angles...")
    robot.plot_multiple_views(joint_angles, 'robot_multiple_views.png')
    print("   Saved: robot_multiple_views.png")
    
    # ========================================================================
    # 3. Trajectory Visualization
    # ========================================================================
    print("\n3. Generating trajectory visualization...")
    start_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end_angles = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    
    angles, velocities, accelerations = robot.generate_trajectory_joint_space(
        start_angles, end_angles, duration=3.0, dt=0.05
    )
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    robot.plot_trajectory(angles, ax=ax)
    plt.savefig('robot_trajectory.png', dpi=150, bbox_inches='tight')
    print("   Saved: robot_trajectory.png")
    plt.close()
    
    # ========================================================================
    # 4. Workspace Visualization
    # ========================================================================
    print("\n4. Generating workspace visualization...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    robot.plot_workspace(n_samples=2000, ax=ax)
    plt.savefig('robot_workspace.png', dpi=150, bbox_inches='tight')
    print("   Saved: robot_workspace.png")
    plt.close()
    
    # ========================================================================
    # 5. Multiple Configurations Comparison
    # ========================================================================
    print("\n5. Generating multiple configurations comparison...")
    fig = plt.figure(figsize=(16, 12))
    
    configs = [
        (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "Home Position"),
        (np.array([0.5, 0.3, -0.2, 0.4, 0.6, 0.1]), "Configuration 1"),
        (np.array([-0.3, 0.5, 0.2, -0.4, 0.3, -0.2]), "Configuration 2"),
        (np.array([0.2, -0.4, 0.5, 0.3, -0.2, 0.4]), "Configuration 3"),
    ]
    
    for idx, (angles, title) in enumerate(configs):
        ax = fig.add_subplot(2, 2, idx+1, projection='3d')
        robot.plot_robot(angles, ax)
        ax.set_title(title, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('robot_configurations.png', dpi=150, bbox_inches='tight')
    print("   Saved: robot_configurations.png")
    plt.close()
    
    # ========================================================================
    # 6. Trajectory Animation (Interactive)
    # ========================================================================
    print("\n6. Creating trajectory animation...")
    print("   (This will open an interactive window)")
    
    # Create a smooth trajectory
    angles, _, _ = robot.generate_trajectory_quintic(
        start_angles, end_angles,
        np.zeros(6), np.zeros(6),
        duration=5.0, dt=0.1
    )
    
    # Subsample for smoother animation
    angles_subset = angles[::2]  # Every other frame
    
    try:
        anim = robot.animate_trajectory(angles_subset, interval=100, save_gif=False)
        print("   Animation window opened. Close to continue...")
        plt.show()
    except Exception as e:
        print(f"   Could not create animation: {e}")
    
    # ========================================================================
    # 7. End-Effector Path Visualization
    # ========================================================================
    print("\n7. Generating end-effector path visualization...")
    angles, _, _ = robot.generate_trajectory_joint_space(
        start_angles, end_angles, duration=4.0, dt=0.05
    )
    
    # Compute end-effector positions
    ee_positions = np.zeros((len(angles), 3))
    for i, q in enumerate(angles):
        pos, _ = robot.forward_kinematics(q)
        ee_positions[i] = pos
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
           'b-', linewidth=3, alpha=0.7, label='End-Effector Path')
    
    # Color by time
    scatter = ax.scatter(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
                        c=np.arange(len(ee_positions)), cmap='viridis', 
                        s=50, alpha=0.8, label='Trajectory Points')
    
    # Plot start and end
    ax.scatter(ee_positions[0, 0], ee_positions[0, 1], ee_positions[0, 2],
              c='green', s=300, marker='o', label='Start', zorder=10, edgecolors='black')
    ax.scatter(ee_positions[-1, 0], ee_positions[-1, 1], ee_positions[-1, 2],
              c='red', s=300, marker='s', label='End', zorder=10, edgecolors='black')
    
    # Plot robot at start and end
    robot.plot_robot(start_angles, ax)
    robot.plot_robot(end_angles, ax)
    
    plt.colorbar(scatter, ax=ax, label='Time Step')
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax.set_title('End-Effector Trajectory Path', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    max_range = 0.8
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-0.1, max_range])
    ax.set_box_aspect([1, 1, 1])
    
    plt.savefig('robot_ee_path.png', dpi=150, bbox_inches='tight')
    print("   Saved: robot_ee_path.png")
    plt.close()
    
    print("\n" + "=" * 60)
    print("Visualization Demo Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - robot_3d_enhanced.png")
    print("  - robot_multiple_views.png")
    print("  - robot_trajectory.png")
    print("  - robot_workspace.png")
    print("  - robot_configurations.png")
    print("  - robot_ee_path.png")
    print("\nAll visualizations saved successfully!")


if __name__ == "__main__":
    main()
