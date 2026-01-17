"""
Vision-Guided Robotics Integration
==================================

This module integrates the Vesuvius Surface Detection Model (Computer Vision)
with the KUKA LBR iisy 3 R760 Robot (Robotics) for vision-guided manipulation.

Integration Use Cases:
1. Surface Detection & Robot Positioning
2. Quality Inspection with Robot Manipulation
3. 3D Volume Analysis for Robot Path Planning
"""

import numpy as np
import sys
from pathlib import Path

# Add Computer Vision module to path
cv_path = Path(__file__).parent / "Computer Vision"
cv_path_str = str(cv_path)
if cv_path_str not in sys.path:
    sys.path.insert(0, cv_path_str)

# Import robotics module
from robotics import KUKALBRiisy3R760

# Import computer vision modules
# Note: Path is added to sys.path above to enable these imports
try:
    from vesuvius_model import UNet3D  # type: ignore[import-untyped]
    from inference import load_model, predict_volume  # type: ignore[import-untyped]
    CV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Computer Vision modules not available: {e}")
    CV_AVAILABLE = False
    UNet3D = None  # type: ignore[assignment]
    load_model = None  # type: ignore[assignment]
    predict_volume = None  # type: ignore[assignment]


class VisionGuidedRobot:
    """
    Integrated system combining Computer Vision and Robotics.
    
    This class demonstrates how the Vesuvius surface detection model
    can guide robot manipulation tasks.
    """
    
    def __init__(self, cv_model_path=None):
        """
        Initialize the integrated system.
        
        Args:
            cv_model_path: Path to trained CV model checkpoint (optional)
        """
        # Initialize robot
        self.robot = KUKALBRiisy3R760()
        
        # Initialize computer vision model
        self.cv_model = None
        if CV_AVAILABLE:
            if cv_model_path:
                try:
                    self.cv_model = load_model(cv_model_path, device='cpu')
                    print("[OK] Computer Vision model loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load CV model: {e}")
                    print("Using untrained model for demonstration")
                    self.cv_model = UNet3D(in_ch=1, base_ch=32, out_ch=1)
            else:
                # Use untrained model for demonstration
                self.cv_model = UNet3D(in_ch=1, base_ch=32, out_ch=1)
                print("Using untrained CV model for demonstration")
        else:
            print("Computer Vision module not available")
    
    def detect_surface_regions(self, volume_data):
        """
        Detect surface/ink regions in 3D volume using CV model.
        
        Args:
            volume_data: 3D numpy array [D, H, W] or [C, D, H, W]
            
        Returns:
            predictions: Binary predictions of surface regions
            probabilities: Probability map
        """
        if not CV_AVAILABLE or self.cv_model is None:
            raise RuntimeError("Computer Vision model not available")
        
        predictions, probabilities = predict_volume(
            self.cv_model, volume_data, device='cpu', threshold=0.5
        )
        
        return predictions, probabilities
    
    def find_surface_centroid(self, predictions):
        """
        Find the centroid of detected surface regions.
        
        Args:
            predictions: Binary prediction array [D, H, W]
            
        Returns:
            centroid: 3D position (x, y, z) in volume coordinates
        """
        # Find all positive voxels
        positive_voxels = np.argwhere(predictions > 0)
        
        if len(positive_voxels) == 0:
            return None
        
        # Calculate centroid
        centroid = np.mean(positive_voxels, axis=0)
        
        # Convert from volume coordinates to robot coordinates
        # This is a simplified mapping - adjust based on your setup
        # Assuming volume is in mm and needs scaling/translation
        robot_position = centroid * 0.001  # Convert mm to meters (example)
        
        return robot_position
    
    def plan_robot_path_to_surface(self, target_position, approach_distance=0.1):
        """
        Plan robot trajectory to approach detected surface.
        
        Args:
            target_position: 3D target position in robot frame (m)
            approach_distance: Distance to maintain from surface (m)
            
        Returns:
            trajectory: Array of joint angles for trajectory
        """
        # Get current robot position
        current_angles = np.zeros(6)  # Start from home position
        current_pos, _ = self.robot.forward_kinematics(current_angles)
        
        # Calculate approach position (offset from target)
        # Approach from above (negative z direction)
        approach_pos = target_position.copy()
        approach_pos[2] += approach_distance
        
        # Plan trajectory: current -> approach -> target
        # Step 1: Move to approach position
        ik_approach, success1 = self.robot.inverse_kinematics_numerical(
            approach_pos, 
            target_orient=np.eye(3),
            initial_guess=current_angles
        )
        
        if not success1:
            print("Warning: Could not find IK solution for approach position")
            ik_approach = current_angles
        
        # Step 2: Move to target position
        ik_target, success2 = self.robot.inverse_kinematics_numerical(
            target_position,
            target_orient=np.eye(3),
            initial_guess=ik_approach
        )
        
        if not success2:
            print("Warning: Could not find IK solution for target position")
            ik_target = ik_approach
        
        # Generate smooth trajectory
        angles1, _, _ = self.robot.generate_trajectory_joint_space(
            current_angles, ik_approach, duration=2.0, dt=0.01
        )
        
        angles2, _, _ = self.robot.generate_trajectory_joint_space(
            ik_approach, ik_target, duration=1.0, dt=0.01
        )
        
        # Combine trajectories
        trajectory = np.vstack([angles1, angles2])
        
        return trajectory, success1 and success2
    
    def inspect_surface_with_robot(self, volume_data, inspection_positions=None):
        """
        Use robot to inspect detected surface regions.
        
        Args:
            volume_data: 3D volume data
            inspection_positions: Optional list of specific positions to inspect
            
        Returns:
            inspection_results: Dictionary with inspection data
        """
        # Detect surface regions
        print("Detecting surface regions...")
        predictions, probabilities = self.detect_surface_regions(volume_data)
        
        # Find surface centroid
        surface_centroid = self.find_surface_centroid(predictions)
        
        if surface_centroid is None:
            print("No surface regions detected")
            return None
        
        print(f"Surface centroid detected at: {surface_centroid} (m)")
        
        # Plan robot path
        print("Planning robot trajectory...")
        trajectory, success = self.plan_robot_path_to_surface(surface_centroid)
        
        if not success:
            print("Warning: Trajectory planning had issues")
        
        # Compute end-effector positions along trajectory
        ee_positions = []
        for angles in trajectory:
            pos, _ = self.robot.forward_kinematics(angles)
            ee_positions.append(pos)
        
        ee_positions = np.array(ee_positions)
        
        # Prepare results
        results = {
            'surface_centroid': surface_centroid,
            'trajectory': trajectory,
            'ee_positions': ee_positions,
            'predictions': predictions,
            'probabilities': probabilities,
            'success': success
        }
        
        return results
    
    def visualize_integrated_system(self, volume_data, results=None):
        """
        Visualize the integrated vision-robotics system.
        
        Args:
            volume_data: 3D volume data
            results: Inspection results from inspect_surface_with_robot
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        if results is None:
            results = self.inspect_surface_with_robot(volume_data)
        
        if results is None:
            print("No results to visualize")
            return
        
        fig = plt.figure(figsize=(16, 6))
        
        # Plot 1: Robot trajectory
        ax1 = fig.add_subplot(131, projection='3d')
        self.robot.plot_trajectory(results['trajectory'], 
                                   trajectory_positions=results['ee_positions'],
                                   ax=ax1)
        ax1.scatter(results['surface_centroid'][0],
                   results['surface_centroid'][1],
                   results['surface_centroid'][2],
                   c='yellow', s=500, marker='*', 
                   label='Detected Surface', zorder=10)
        ax1.set_title('Robot Trajectory to Surface', fontweight='bold')
        ax1.legend()
        
        # Plot 2: Surface detection visualization (2D slice)
        ax2 = fig.add_subplot(132)
        if results['predictions'].ndim == 3:
            mid_slice = results['predictions'].shape[0] // 2
            ax2.imshow(results['predictions'][mid_slice], cmap='hot', alpha=0.7)
            ax2.set_title('Surface Detection (Mid Slice)', fontweight='bold')
            ax2.set_xlabel('Width')
            ax2.set_ylabel('Height')
        
        # Plot 3: Robot at target position
        ax3 = fig.add_subplot(133, projection='3d')
        final_angles = results['trajectory'][-1]
        self.robot.plot_robot(final_angles, ax3)
        ax3.scatter(results['surface_centroid'][0],
                   results['surface_centroid'][1],
                   results['surface_centroid'][2],
                   c='yellow', s=500, marker='*', 
                   label='Target Surface', zorder=10)
        ax3.set_title('Robot at Target Position', fontweight='bold')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig('vision_guided_robotics.png', dpi=150, bbox_inches='tight')
        print("Visualization saved to 'vision_guided_robotics.png'")
        plt.close()


def example_integration():
    """Example demonstrating the integrated system"""
    
    print("=" * 70)
    print("Vision-Guided Robotics Integration Demo")
    print("=" * 70)
    
    # Initialize integrated system
    print("\n1. Initializing integrated system...")
    system = VisionGuidedRobot()
    
    # Create dummy 3D volume data (simulating CT scan)
    print("\n2. Generating dummy 3D volume data...")
    volume = np.random.randn(64, 64, 64).astype(np.float32)
    # Add some structure to make it more interesting
    volume[20:40, 20:40, 20:40] += 2.0  # Simulated surface region
    
    # Run integrated inspection
    print("\n3. Running vision-guided robot inspection...")
    results = system.inspect_surface_with_robot(volume)
    
    if results:
        print(f"\n[OK] Surface detected at: {results['surface_centroid']} (m)")
        print(f"[OK] Trajectory generated: {len(results['trajectory'])} waypoints")
        print(f"[OK] Success: {results['success']}")
        
        # Visualize
        print("\n4. Generating visualization...")
        system.visualize_integrated_system(volume, results)
        
        print("\n" + "=" * 70)
        print("Integration Demo Complete!")
        print("=" * 70)
    else:
        print("\n[ERROR] Inspection failed - no surface detected")


if __name__ == "__main__":
    example_integration()
