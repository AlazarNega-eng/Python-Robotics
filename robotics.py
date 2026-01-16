"""
KUKA LBR iisy 3 R760 6 DOF Robot Simulation and Implementation
==============================================================

This module provides comprehensive robotics functionality including:
- Forward Kinematics
- Inverse Kinematics
- Jacobian Computation
- Dynamics (Mass, Coriolis, Gravity)
- Trajectory Generation
- Control (Low-Level and High-Level)

Author: Robotics Simulation
Robot: KUKA LBR iisy 3 R760
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class KUKALBRiisy3R760:
    """
    KUKA LBR iisy 3 R760 6 DOF Robot Model
    
    Specifications:
    - Maximum reach: 760 mm
    - Rated payload: 3 kg
    - Number of axes: 6
    - Pose repeatability: ±0.1 mm
    """
    
    def __init__(self):
        """Initialize robot with MDH parameters"""
        # Modified Denavit-Hartenberg Parameters
        # Format: [alpha_i-1 (rad), a_i-1 (m), d_i (m)]
        self.mdh_params = np.array([
            [np.pi/2,  0.000,  0.340],  # Joint 1
            [0.0,      0.000,  0.000],  # Joint 2
            [0.0,      0.000,  0.000],  # Joint 3
            [np.pi/2,  0.000,  0.300],  # Joint 4
            [-np.pi/2, 0.000,  0.000],  # Joint 5
            [0.0,      0.000,  0.120],  # Joint 6
        ])
        
        # Joint limits (from datasheet)
        self.joint_limits = np.array([
            [-np.pi, np.pi],           # A1: ±185°
            [-230*np.pi/180, 50*np.pi/180],  # A2: -230° / 50°
            [-np.pi, np.pi],           # A3: ±150° (using ±180° as approximation)
            [-np.pi, np.pi],           # A4: ±175° (using ±180° as approximation)
            [-np.pi, np.pi],           # A5: ±110° (using ±180° as approximation)
            [-np.pi, np.pi],           # A6: ±220° (using ±180° as approximation)
        ])
        
        # Maximum joint velocities (rad/s)
        self.max_velocities = np.array([200, 200, 200, 300, 300, 400]) * np.pi / 180
        
        # Robot mass and inertia (approximated - adjust based on actual specs)
        self.link_masses = np.array([5.0, 4.0, 3.5, 3.0, 2.5, 1.8])  # kg (approximated)
        self.total_mass = 22.8  # kg from datasheet
        
        # Link lengths for visualization
        self.link_lengths = np.array([0.340, 0.0, 0.0, 0.300, 0.0, 0.120])
        
        # Gravity vector (m/s^2)
        self.gravity = np.array([0, 0, -9.81])
        
    # ============================================================================
    # I. FORWARD KINEMATICS
    # ============================================================================
    
    def mdhtransform(self, alpha: float, a: float, d: float, theta: float) -> np.ndarray:
        """
        Compute Modified Denavit-Hartenberg transformation matrix
        
        Args:
            alpha: Twist angle (rad)
            a: Link length (m)
            d: Link offset (m)
            theta: Joint angle (rad)
            
        Returns:
            4x4 homogeneous transformation matrix
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        T = np.array([
            [cos_theta, -sin_theta, 0, a],
            [sin_theta*cos_alpha, cos_theta*cos_alpha, -sin_alpha, -sin_alpha*d],
            [sin_theta*sin_alpha, cos_theta*sin_alpha, cos_alpha, cos_alpha*d],
            [0, 0, 0, 1]
        ])
        
        return T
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics: joint angles -> end-effector pose
        
        Args:
            joint_angles: Array of 6 joint angles (rad)
            
        Returns:
            position: 3D position of end-effector (m)
            orientation: 3x3 rotation matrix
        """
        if len(joint_angles) != 6:
            raise ValueError("Joint angles must have 6 elements")
        
        # Base transformation (identity)
        T = np.eye(4)
        
        # Compute transformation for each joint
        for i in range(6):
            alpha, a, d = self.mdh_params[i]
            theta = joint_angles[i]
            T_i = self.mdhtransform(alpha, a, d, theta)
            T = T @ T_i
        
        # Extract position and orientation
        position = T[:3, 3]
        orientation = T[:3, :3]
        
        return position, orientation
    
    def forward_kinematics_all_joints(self, joint_angles: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute forward kinematics for all intermediate frames
        
        Args:
            joint_angles: Array of 6 joint angles (rad)
            
        Returns:
            List of (position, orientation) tuples for each frame
        """
        frames = []
        T = np.eye(4)
        
        for i in range(6):
            alpha, a, d = self.mdh_params[i]
            theta = joint_angles[i]
            T_i = self.mdhtransform(alpha, a, d, theta)
            T = T @ T_i
            
            position = T[:3, 3]
            orientation = T[:3, :3]
            frames.append((position.copy(), orientation.copy()))
        
        return frames
    
    # ============================================================================
    # II. INVERSE KINEMATICS
    # ============================================================================
    
    def inverse_kinematics_geometric(self, target_pos: np.ndarray, 
                                     target_orient: Optional[np.ndarray] = None,
                                     initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Geometric inverse kinematics (simplified for 6 DOF)
        Uses numerical optimization as analytical solution is complex
        
        Args:
            target_pos: Desired 3D position (m)
            target_orient: Desired 3x3 rotation matrix (optional)
            initial_guess: Initial joint angles for optimization (rad)
            
        Returns:
            Joint angles (rad)
        """
        if initial_guess is None:
            initial_guess = np.zeros(6)
        
        if target_orient is None:
            # If no orientation specified, use identity
            target_orient = np.eye(3)
        
        def cost_function(q):
            pos, orient = self.forward_kinematics(q)
            
            # Position error
            pos_error = np.linalg.norm(pos - target_pos)
            
            # Orientation error (using rotation matrix difference)
            orient_error = np.linalg.norm(orient - target_orient)
            
            # Joint limit penalty
            limit_penalty = 0
            for i in range(6):
                if q[i] < self.joint_limits[i, 0] or q[i] > self.joint_limits[i, 1]:
                    limit_penalty += 1000
            
            return pos_error + 0.1 * orient_error + limit_penalty
        
        # Optimize
        result = minimize(cost_function, initial_guess, method='BFGS')
        
        # Clamp to joint limits
        q_solution = np.clip(result.x, 
                            self.joint_limits[:, 0], 
                            self.joint_limits[:, 1])
        
        return q_solution
    
    def inverse_kinematics_numerical(self, target_pos: np.ndarray,
                                    target_orient: Optional[np.ndarray] = None,
                                    initial_guess: Optional[np.ndarray] = None,
                                    max_iterations: int = 100,
                                    tolerance: float = 1e-6) -> Tuple[np.ndarray, bool]:
        """
        Numerical inverse kinematics using Jacobian-based iterative method
        
        Args:
            target_pos: Desired 3D position (m)
            target_orient: Desired 3x3 rotation matrix (optional)
            initial_guess: Initial joint angles (rad)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Joint angles (rad), success flag
        """
        if initial_guess is None:
            q = np.zeros(6)
        else:
            q = initial_guess.copy()
        
        if target_orient is None:
            target_orient = np.eye(3)
        
        for iteration in range(max_iterations):
            # Current pose
            pos, orient = self.forward_kinematics(q)
            
            # Position error
            pos_error = target_pos - pos
            
            # Orientation error (convert to axis-angle representation)
            R_error = target_orient @ orient.T
            angle_error = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
            if angle_error > tolerance:
                axis_error = np.array([
                    R_error[2, 1] - R_error[1, 2],
                    R_error[0, 2] - R_error[2, 0],
                    R_error[1, 0] - R_error[0, 1]
                ]) / (2 * np.sin(angle_error))
                orient_error = angle_error * axis_error
            else:
                orient_error = np.zeros(3)
            
            # Combined error
            error = np.concatenate([pos_error, orient_error])
            
            # Check convergence
            if np.linalg.norm(error) < tolerance:
                return q, True
            
            # Compute Jacobian
            J = self.jacobian(q)
            
            # Damped least squares
            damping = 0.01
            dq = np.linalg.solve(J.T @ J + damping * np.eye(6), J.T @ error)
            
            # Update joint angles
            q = q + dq
            
            # Clamp to joint limits
            q = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])
        
        return q, False
    
    # ============================================================================
    # III. JACOBIAN
    # ============================================================================
    
    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute geometric Jacobian matrix (6x6)
        
        Args:
            joint_angles: Array of 6 joint angles (rad)
            
        Returns:
            6x6 Jacobian matrix [v; w] where v is linear velocity and w is angular velocity
        """
        # Get all frame positions and z-axes
        frames = self.forward_kinematics_all_joints(joint_angles)
        end_pos, end_orient = self.forward_kinematics(joint_angles)
        
        J = np.zeros((6, 6))
        
        # Base transformation
        T = np.eye(4)
        z_axes = []
        origins = [np.array([0, 0, 0])]
        
        for i in range(6):
            alpha, a, d = self.mdh_params[i]
            theta = joint_angles[i]
            T_i = self.mdhtransform(alpha, a, d, theta)
            T = T @ T_i
            
            # z-axis in base frame
            z_axis = T[:3, 2]
            z_axes.append(z_axis)
            
            # Origin position
            origin = T[:3, 3]
            origins.append(origin)
        
        # Compute Jacobian columns
        for i in range(6):
            # Angular velocity component (always z-axis for revolute joints)
            J[3:6, i] = z_axes[i]
            
            # Linear velocity component
            r = end_pos - origins[i]
            J[0:3, i] = np.cross(z_axes[i], r)
        
        return J
    
    def jacobian_dot(self, joint_angles: np.ndarray, joint_velocities: np.ndarray) -> np.ndarray:
        """
        Compute time derivative of Jacobian
        
        Args:
            joint_angles: Array of 6 joint angles (rad)
            joint_velocities: Array of 6 joint velocities (rad/s)
            
        Returns:
            6x6 Jacobian derivative matrix
        """
        h = 1e-6
        J = self.jacobian(joint_angles)
        J_dot = np.zeros((6, 6))
        
        for i in range(6):
            q_perturbed = joint_angles.copy()
            q_perturbed[i] += h
            J_perturbed = self.jacobian(q_perturbed)
            J_dot[:, i] = (J_perturbed - J) @ joint_velocities / h
        
        return J_dot
    
    # ============================================================================
    # IV. DYNAMICS
    # ============================================================================
    
    def compute_mass_matrix(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute mass matrix (inertia matrix) M(q)
        
        Args:
            joint_angles: Array of 6 joint angles (rad)
            
        Returns:
            6x6 mass matrix
        """
        # Simplified mass matrix computation
        # For full implementation, would need link inertias
        M = np.zeros((6, 6))
        
        # Get Jacobian for each link
        frames = self.forward_kinematics_all_joints(joint_angles)
        
        for i in range(6):
            # Compute Jacobian for link i center of mass
            # Simplified: assume center of mass at link center
            J_i = self.jacobian(joint_angles)
            
            # Simplified inertia (would need actual link inertias)
            I_i = np.eye(3) * 0.1  # Approximate inertia
            
            # Mass matrix contribution
            M += self.link_masses[i] * J_i[:3, :].T @ J_i[:3, :]
            M += J_i[3:, :].T @ I_i @ J_i[3:, :]
        
        # Add diagonal terms for joint inertia
        joint_inertias = np.array([0.1, 0.08, 0.06, 0.05, 0.04, 0.03])
        M += np.diag(joint_inertias)
        
        return M
    
    def compute_coriolis_matrix(self, joint_angles: np.ndarray, 
                               joint_velocities: np.ndarray) -> np.ndarray:
        """
        Compute Coriolis and centrifugal matrix C(q, q_dot)
        
        Args:
            joint_angles: Array of 6 joint angles (rad)
            joint_velocities: Array of 6 joint velocities (rad/s)
            
        Returns:
            6x6 Coriolis matrix
        """
        M = self.compute_mass_matrix(joint_angles)
        C = np.zeros((6, 6))
        
        # Compute using Christoffel symbols
        h = 1e-6
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    # Partial derivative of M_ij with respect to q_k
                    q_perturbed = joint_angles.copy()
                    q_perturbed[k] += h
                    M_perturbed = self.compute_mass_matrix(q_perturbed)
                    
                    dM_ij_dqk = (M_perturbed[i, j] - M[i, j]) / h
                    
                    # Christoffel symbol
                    c_ijk = 0.5 * (dM_ij_dqk + 
                                  (M_perturbed[i, k] - M[i, k]) / h - 
                                  (M_perturbed[j, k] - M[j, k]) / h)
                    
                    C[i, j] += c_ijk * joint_velocities[k]
        
        return C
    
    def compute_gravity_vector(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute gravity vector G(q)
        
        Args:
            joint_angles: Array of 6 joint angles (rad)
            
        Returns:
            6x1 gravity vector
        """
        G = np.zeros(6)
        
        # Get center of mass positions for each link
        frames = self.forward_kinematics_all_joints(joint_angles)
        
        for i in range(6):
            # Simplified: assume center of mass at frame origin
            if i < len(frames):
                com_pos = frames[i][0]
            else:
                com_pos, _ = self.forward_kinematics(joint_angles)
            
            # Compute Jacobian for center of mass
            J_i = self.jacobian(joint_angles)
            
            # Gravity contribution
            G += self.link_masses[i] * J_i[:3, :].T @ self.gravity
        
        return G
    
    def compute_dynamics(self, joint_angles: np.ndarray, 
                        joint_velocities: np.ndarray,
                        joint_accelerations: np.ndarray) -> np.ndarray:
        """
        Compute joint torques using Euler-Lagrange equations
        
        tau = M(q) * q_ddot + C(q, q_dot) * q_dot + G(q)
        
        Args:
            joint_angles: Array of 6 joint angles (rad)
            joint_velocities: Array of 6 joint velocities (rad/s)
            joint_accelerations: Array of 6 joint accelerations (rad/s^2)
            
        Returns:
            Joint torques (N*m)
        """
        M = self.compute_mass_matrix(joint_angles)
        C = self.compute_coriolis_matrix(joint_angles, joint_velocities)
        G = self.compute_gravity_vector(joint_angles)
        
        tau = M @ joint_accelerations + C @ joint_velocities + G
        
        return tau
    
    # ============================================================================
    # V. TRAJECTORY GENERATION
    # ============================================================================
    
    def generate_trajectory_linear(self, start_pos: np.ndarray, end_pos: np.ndarray,
                                  duration: float, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate linear trajectory in Cartesian space
        
        Args:
            start_pos: Starting 3D position (m)
            end_pos: Ending 3D position (m)
            duration: Trajectory duration (s)
            dt: Time step (s)
            
        Returns:
            positions: Array of positions
            times: Array of time points
        """
        times = np.arange(0, duration + dt, dt)
        n_points = len(times)
        
        positions = np.zeros((n_points, 3))
        
        for i, t in enumerate(times):
            s = t / duration  # Normalized time [0, 1]
            # Cubic polynomial for smooth motion
            s_smooth = 3 * s**2 - 2 * s**3
            positions[i] = start_pos + s_smooth * (end_pos - start_pos)
        
        return positions, times
    
    def generate_trajectory_joint_space(self, start_angles: np.ndarray, 
                                       end_angles: np.ndarray,
                                       duration: float, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate trajectory in joint space using cubic polynomials
        
        Args:
            start_angles: Starting joint angles (rad)
            end_angles: Ending joint angles (rad)
            duration: Trajectory duration (s)
            dt: Time step (s)
            
        Returns:
            angles: Array of joint angles
            velocities: Array of joint velocities
            accelerations: Array of joint accelerations
        """
        times = np.arange(0, duration + dt, dt)
        n_points = len(times)
        n_joints = len(start_angles)
        
        angles = np.zeros((n_points, n_joints))
        velocities = np.zeros((n_points, n_joints))
        accelerations = np.zeros((n_points, n_joints))
        
        for j in range(n_joints):
            q0 = start_angles[j]
            qf = end_angles[j]
            
            # Cubic polynomial coefficients
            a0 = q0
            a1 = 0  # Start with zero velocity
            a2 = 3 * (qf - q0) / duration**2
            a3 = -2 * (qf - q0) / duration**3
            
            for i, t in enumerate(times):
                angles[i, j] = a0 + a1*t + a2*t**2 + a3*t**3
                velocities[i, j] = a1 + 2*a2*t + 3*a3*t**2
                accelerations[i, j] = 2*a2 + 6*a3*t
        
        return angles, velocities, accelerations
    
    def generate_trajectory_quintic(self, start_angles: np.ndarray,
                                   end_angles: np.ndarray,
                                   start_velocities: np.ndarray,
                                   end_velocities: np.ndarray,
                                   duration: float, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate quintic polynomial trajectory in joint space
        
        Args:
            start_angles: Starting joint angles (rad)
            end_angles: Ending joint angles (rad)
            start_velocities: Starting joint velocities (rad/s)
            end_velocities: Ending joint velocities (rad/s)
            duration: Trajectory duration (s)
            dt: Time step (s)
            
        Returns:
            angles: Array of joint angles
            velocities: Array of joint velocities
            accelerations: Array of joint accelerations
        """
        times = np.arange(0, duration + dt, dt)
        n_points = len(times)
        n_joints = len(start_angles)
        
        angles = np.zeros((n_points, n_joints))
        velocities = np.zeros((n_points, n_joints))
        accelerations = np.zeros((n_points, n_joints))
        
        for j in range(n_joints):
            q0 = start_angles[j]
            qf = end_angles[j]
            v0 = start_velocities[j]
            vf = end_velocities[j]
            
            # Quintic polynomial coefficients
            T = duration
            a0 = q0
            a1 = v0
            a2 = 0  # Start with zero acceleration
            a3 = (20*(qf - q0) - (8*vf + 12*v0)*T) / (2*T**3)
            a4 = (30*(q0 - qf) + (14*vf + 16*v0)*T) / (T**4)
            a5 = (12*(qf - q0) - (6*vf + 6*v0)*T) / (T**5)
            
            for i, t in enumerate(times):
                angles[i, j] = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
                velocities[i, j] = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
                accelerations[i, j] = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
        
        return angles, velocities, accelerations
    
    # ============================================================================
    # VI. CONTROL
    # ============================================================================
    
    class PIDController:
        """Low-level PID controller for individual joints"""
        
        def __init__(self, kp: float, ki: float, kd: float, dt: float = 0.01):
            """
            Initialize PID controller
            
            Args:
                kp: Proportional gain
                ki: Integral gain
                kd: Derivative gain
                dt: Time step (s)
            """
            self.kp = kp
            self.ki = ki
            self.kd = kd
            self.dt = dt
            self.integral = 0.0
            self.prev_error = 0.0
        
        def compute(self, error: float) -> float:
            """
            Compute control output
            
            Args:
                error: Current error
                
            Returns:
                Control output
            """
            self.integral += error * self.dt
            derivative = (error - self.prev_error) / self.dt
            
            output = self.kp * error + self.ki * self.integral + self.kd * derivative
            
            self.prev_error = error
            
            return output
        
        def reset(self):
            """Reset controller state"""
            self.integral = 0.0
            self.prev_error = 0.0
    
    def compute_control_torque_pid(self, desired_angles: np.ndarray,
                                   current_angles: np.ndarray,
                                   desired_velocities: np.ndarray,
                                   current_velocities: np.ndarray,
                                   controllers: List[PIDController]) -> np.ndarray:
        """
        Compute control torques using PID controllers (low-level)
        
        Args:
            desired_angles: Desired joint angles (rad)
            current_angles: Current joint angles (rad)
            desired_velocities: Desired joint velocities (rad/s)
            current_velocities: Current joint velocities (rad/s)
            controllers: List of 6 PID controllers
            
        Returns:
            Control torques (N*m)
        """
        torques = np.zeros(6)
        
        for i in range(6):
            # Position error
            pos_error = desired_angles[i] - current_angles[i]
            
            # Velocity error
            vel_error = desired_velocities[i] - current_velocities[i]
            
            # Combined error (can be tuned)
            error = pos_error + 0.1 * vel_error
            
            torques[i] = controllers[i].compute(error)
        
        return torques
    
    def compute_control_torque_computed_torque(self, desired_angles: np.ndarray,
                                              current_angles: np.ndarray,
                                              desired_velocities: np.ndarray,
                                              current_velocities: np.ndarray,
                                              desired_accelerations: np.ndarray,
                                              kp: np.ndarray, kd: np.ndarray) -> np.ndarray:
        """
        Computed torque control (high-level)
        
        tau = M(q) * (q_ddot_d + Kp*e + Kd*e_dot) + C(q, q_dot) * q_dot + G(q)
        
        Args:
            desired_angles: Desired joint angles (rad)
            current_angles: Current joint angles (rad)
            desired_velocities: Desired joint velocities (rad/s)
            current_velocities: Current joint velocities (rad/s)
            desired_accelerations: Desired joint accelerations (rad/s^2)
            kp: Proportional gain matrix (6x6 or 6x1)
            kd: Derivative gain matrix (6x6 or 6x1)
            
        Returns:
            Control torques (N*m)
        """
        # Errors
        e = desired_angles - current_angles
        e_dot = desired_velocities - current_velocities
        
        # Convert to diagonal matrices if vectors
        if kp.ndim == 1:
            Kp = np.diag(kp)
        else:
            Kp = kp
        
        if kd.ndim == 1:
            Kd = np.diag(kd)
        else:
            Kd = kd
        
        # Computed acceleration
        q_ddot_computed = desired_accelerations + Kp @ e + Kd @ e_dot
        
        # Dynamics
        M = self.compute_mass_matrix(current_angles)
        C = self.compute_coriolis_matrix(current_angles, current_velocities)
        G = self.compute_gravity_vector(current_angles)
        
        # Control torque
        tau = M @ q_ddot_computed + C @ current_velocities + G
        
        return tau
    
    def compute_control_torque_operational_space(self, desired_pos: np.ndarray,
                                                 current_pos: np.ndarray,
                                                 desired_orient: np.ndarray,
                                                 current_orient: np.ndarray,
                                                 current_angles: np.ndarray,
                                                 kp: float = 100.0, kd: float = 20.0) -> np.ndarray:
        """
        Operational space control (high-level)
        
        Args:
            desired_pos: Desired end-effector position (m)
            current_pos: Current end-effector position (m)
            desired_orient: Desired end-effector orientation (3x3)
            current_orient: Current end-effector orientation (3x3)
            current_angles: Current joint angles (rad)
            kp: Position gain
            kd: Damping gain
            
        Returns:
            Control torques (N*m)
        """
        # Position error
        pos_error = desired_pos - current_pos
        
        # Orientation error
        R_error = desired_orient @ current_orient.T
        angle_error = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
        if angle_error > 1e-6:
            axis_error = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1]
            ]) / (2 * np.sin(angle_error))
            orient_error = angle_error * axis_error
        else:
            orient_error = np.zeros(3)
        
        # Combined error
        error = np.concatenate([pos_error, orient_error])
        
        # Desired force in operational space
        F_desired = kp * error  # Simplified (no velocity feedback)
        
        # Jacobian
        J = self.jacobian(current_angles)
        
        # Transform to joint space
        tau = J.T @ F_desired
        
        return tau
    
    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    
    def plot_robot(self, joint_angles: np.ndarray, ax: Optional[plt.Axes] = None, 
                   show_frames: bool = False, color: str = 'blue') -> plt.Axes:
        """
        Plot robot configuration with enhanced 3D visualization
        
        Args:
            joint_angles: Array of 6 joint angles (rad)
            ax: Matplotlib 3D axes (optional)
            show_frames: Whether to show coordinate frames
            color: Color scheme for the robot
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        # Get all frame positions
        frames = self.forward_kinematics_all_joints(joint_angles)
        
        # Plot base (larger and more visible)
        base_radius = 0.05
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_base = base_radius * np.outer(np.cos(u), np.sin(v))
        y_base = base_radius * np.outer(np.sin(u), np.sin(v))
        z_base = base_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_base, y_base, z_base, color='darkgray', alpha=0.7)
        ax.scatter(0, 0, 0, c='black', s=200, marker='o', label='Base', zorder=10)
        
        # Plot links with different colors
        positions = [np.array([0, 0, 0])]
        for pos, _ in frames:
            positions.append(pos)
        
        positions = np.array(positions)
        
        # Plot links with gradient colors
        colors_list = plt.cm.viridis(np.linspace(0, 1, len(positions)-1))
        for i in range(len(positions)-1):
            ax.plot([positions[i, 0], positions[i+1, 0]], 
                   [positions[i, 1], positions[i+1, 1]], 
                   [positions[i, 2], positions[i+1, 2]], 
                   color=colors_list[i], linewidth=4, alpha=0.8, zorder=5)
            # Plot joints
            ax.scatter(positions[i+1, 0], positions[i+1, 1], positions[i+1, 2], 
                      c=colors_list[i], s=100, marker='o', edgecolors='black', 
                      linewidths=1.5, zorder=6)
        
        # Plot coordinate frames if requested
        if show_frames:
            frame_length = 0.05
            for i, (pos, orient) in enumerate(frames):
                # X axis (red)
                x_axis = pos + orient[:, 0] * frame_length
                ax.plot([pos[0], x_axis[0]], [pos[1], x_axis[1]], [pos[2], x_axis[2]], 
                       'r-', linewidth=2, alpha=0.7)
                # Y axis (green)
                y_axis = pos + orient[:, 1] * frame_length
                ax.plot([pos[0], y_axis[0]], [pos[1], y_axis[1]], [pos[2], y_axis[2]], 
                       'g-', linewidth=2, alpha=0.7)
                # Z axis (blue)
                z_axis = pos + orient[:, 2] * frame_length
                ax.plot([pos[0], z_axis[0]], [pos[1], z_axis[1]], [pos[2], z_axis[2]], 
                       'b-', linewidth=2, alpha=0.7)
        
        # Plot end-effector with special styling
        end_pos, end_orient = self.forward_kinematics(joint_angles)
        ax.scatter(end_pos[0], end_pos[1], end_pos[2], 
                  c='red', s=300, marker='*', label='End-Effector', 
                  edgecolors='darkred', linewidths=2, zorder=10)
        
        # Plot end-effector coordinate frame
        frame_length = 0.08
        x_ee = end_pos + end_orient[:, 0] * frame_length
        y_ee = end_pos + end_orient[:, 1] * frame_length
        z_ee = end_pos + end_orient[:, 2] * frame_length
        ax.plot([end_pos[0], x_ee[0]], [end_pos[1], x_ee[1]], [end_pos[2], x_ee[2]], 
               'r-', linewidth=3, label='X-axis', alpha=0.8)
        ax.plot([end_pos[0], y_ee[0]], [end_pos[1], y_ee[1]], [end_pos[2], y_ee[2]], 
               'g-', linewidth=3, label='Y-axis', alpha=0.8)
        ax.plot([end_pos[0], z_ee[0]], [end_pos[1], z_ee[1]], [end_pos[2], z_ee[2]], 
               'b-', linewidth=3, label='Z-axis', alpha=0.8)
        
        # Set labels and limits
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
        ax.set_title('KUKA LBR iisy 3 R760 Robot Configuration', fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio and limits
        max_range = 0.8  # Maximum reach is 0.76m
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-0.1, max_range])
        ax.set_box_aspect([1, 1, 1])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper left', fontsize=9)
        
        return ax
    
    def animate_trajectory(self, trajectory_angles: np.ndarray, 
                          interval: int = 50, save_gif: bool = False,
                          filename: str = 'robot_animation.gif') -> None:
        """
        Animate robot following a trajectory
        
        Args:
            trajectory_angles: Array of joint angles for each time step (N x 6)
            interval: Animation interval in milliseconds
            save_gif: Whether to save as GIF
            filename: Output filename for GIF
        """
        from matplotlib.animation import FuncAnimation
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initialize plot
        self.plot_robot(trajectory_angles[0], ax)
        
        def update(frame):
            ax.clear()
            self.plot_robot(trajectory_angles[frame], ax)
            ax.set_title(f'KUKA LBR iisy 3 R760 - Frame {frame}/{len(trajectory_angles)-1}', 
                        fontsize=14, fontweight='bold')
        
        anim = FuncAnimation(fig, update, frames=len(trajectory_angles), 
                           interval=interval, repeat=True, blit=False)
        
        if save_gif:
            try:
                anim.save(filename, writer='pillow', fps=20)
                print(f"Animation saved to '{filename}'")
            except Exception as e:
                print(f"Could not save GIF: {e}")
                print("Showing animation instead...")
        
        plt.tight_layout()
        plt.show()
        return anim
    
    def plot_trajectory(self, trajectory_angles: np.ndarray, 
                       trajectory_positions: Optional[np.ndarray] = None,
                       ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot robot trajectory in 3D space
        
        Args:
            trajectory_angles: Array of joint angles (N x 6)
            trajectory_positions: Optional pre-computed positions (N x 3)
            ax: Matplotlib 3D axes (optional)
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        # Compute positions if not provided
        if trajectory_positions is None:
            trajectory_positions = np.zeros((len(trajectory_angles), 3))
            for i, angles in enumerate(trajectory_angles):
                pos, _ = self.forward_kinematics(angles)
                trajectory_positions[i] = pos
        
        # Plot trajectory path
        ax.plot(trajectory_positions[:, 0], 
               trajectory_positions[:, 1], 
               trajectory_positions[:, 2], 
               'r-', linewidth=2, alpha=0.6, label='Trajectory Path')
        
        # Plot start and end points
        ax.scatter(trajectory_positions[0, 0], 
                  trajectory_positions[0, 1], 
                  trajectory_positions[0, 2], 
                  c='green', s=200, marker='o', label='Start', zorder=10)
        ax.scatter(trajectory_positions[-1, 0], 
                  trajectory_positions[-1, 1], 
                  trajectory_positions[-1, 2], 
                  c='red', s=200, marker='s', label='End', zorder=10)
        
        # Plot robot at key points
        n_keyframes = min(5, len(trajectory_angles))
        keyframe_indices = np.linspace(0, len(trajectory_angles)-1, n_keyframes, dtype=int)
        
        for idx in keyframe_indices:
            self.plot_robot(trajectory_angles[idx], ax)
        
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
        ax.set_title('Robot Trajectory Visualization', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        max_range = 0.8
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-0.1, max_range])
        ax.set_box_aspect([1, 1, 1])
        
        return ax
    
    def plot_workspace(self, n_samples: int = 1000, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Visualize robot workspace by sampling random configurations
        
        Args:
            n_samples: Number of random samples
            ax: Matplotlib 3D axes (optional)
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        # Sample random joint configurations
        workspace_points = []
        for _ in range(n_samples):
            # Random joint angles within limits
            random_angles = np.array([
                np.random.uniform(self.joint_limits[i, 0], self.joint_limits[i, 1])
                for i in range(6)
            ])
            pos, _ = self.forward_kinematics(random_angles)
            workspace_points.append(pos)
        
        workspace_points = np.array(workspace_points)
        
        # Plot workspace points
        ax.scatter(workspace_points[:, 0], 
                  workspace_points[:, 1], 
                  workspace_points[:, 2], 
                  c=workspace_points[:, 2], cmap='viridis', 
                  s=10, alpha=0.5, label='Workspace')
        
        # Plot base
        ax.scatter(0, 0, 0, c='black', s=200, marker='o', label='Base', zorder=10)
        
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
        ax.set_title('KUKA LBR iisy 3 R760 Workspace', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        max_range = 0.8
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-0.1, max_range])
        ax.set_box_aspect([1, 1, 1])
        
        return ax
    
    def plot_multiple_views(self, joint_angles: np.ndarray, 
                           save_path: str = 'robot_multiple_views.png') -> None:
        """
        Plot robot from multiple viewing angles
        
        Args:
            joint_angles: Array of 6 joint angles (rad)
            save_path: Path to save the figure
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Top view
        ax1 = fig.add_subplot(221, projection='3d')
        self.plot_robot(joint_angles, ax1)
        ax1.view_init(elev=90, azim=0)
        ax1.set_title('Top View', fontsize=12, fontweight='bold')
        
        # Front view
        ax2 = fig.add_subplot(222, projection='3d')
        self.plot_robot(joint_angles, ax2)
        ax2.view_init(elev=0, azim=0)
        ax2.set_title('Front View', fontsize=12, fontweight='bold')
        
        # Side view
        ax3 = fig.add_subplot(223, projection='3d')
        self.plot_robot(joint_angles, ax3)
        ax3.view_init(elev=0, azim=90)
        ax3.set_title('Side View', fontsize=12, fontweight='bold')
        
        # Isometric view
        ax4 = fig.add_subplot(224, projection='3d')
        self.plot_robot(joint_angles, ax4)
        ax4.view_init(elev=30, azim=45)
        ax4.set_title('Isometric View', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Multiple views saved to '{save_path}'")
        plt.close()


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def example_usage():
    """Example usage of the KUKA robot simulation"""
    
    # Initialize robot
    robot = KUKALBRiisy3R760()
    
    print("=" * 60)
    print("KUKA LBR iisy 3 R760 Robot Simulation")
    print("=" * 60)
    
    # I. Forward Kinematics
    print("\nI. FORWARD KINEMATICS")
    print("-" * 60)
    joint_angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    position, orientation = robot.forward_kinematics(joint_angles)
    print(f"Joint angles (rad): {joint_angles}")
    print(f"End-effector position (m): {position}")
    print(f"End-effector orientation:\n{orientation}")
    
    # II. Inverse Kinematics
    print("\nII. INVERSE KINEMATICS")
    print("-" * 60)
    # Use a reachable target (from forward kinematics)
    target_pos = position + np.array([0.05, 0.05, 0.05])  # Small offset from current position
    target_orient = orientation
    ik_solution, success = robot.inverse_kinematics_numerical(
        target_pos, target_orient, initial_guess=joint_angles
    )
    print(f"Target position (m): {target_pos}")
    print(f"IK solution (rad): {ik_solution}")
    print(f"Success: {success}")
    
    # Verify IK
    pos_check, orient_check = robot.forward_kinematics(ik_solution)
    print(f"Verification position (m): {pos_check}")
    print(f"Position error (m): {np.linalg.norm(pos_check - target_pos):.6f}")
    
    # III. Jacobian
    print("\nIII. JACOBIAN")
    print("-" * 60)
    J = robot.jacobian(joint_angles)
    print(f"Jacobian matrix shape: {J.shape}")
    print(f"Jacobian condition number: {np.linalg.cond(J)}")
    print(f"Manipulability (det(J*J^T)): {np.sqrt(np.linalg.det(J @ J.T))}")
    
    # IV. Dynamics
    print("\nIV. DYNAMICS")
    print("-" * 60)
    joint_velocities = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    joint_accelerations = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    M = robot.compute_mass_matrix(joint_angles)
    C = robot.compute_coriolis_matrix(joint_angles, joint_velocities)
    G = robot.compute_gravity_vector(joint_angles)
    tau = robot.compute_dynamics(joint_angles, joint_velocities, joint_accelerations)
    
    print(f"Mass matrix shape: {M.shape}")
    print(f"Coriolis matrix shape: {C.shape}")
    print(f"Gravity vector shape: {G.shape}")
    print(f"Required torques (N*m): {tau}")
    
    # V. Trajectory Generation
    print("\nV. TRAJECTORY GENERATION")
    print("-" * 60)
    start_angles = np.zeros(6)
    end_angles = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    angles, velocities, accelerations = robot.generate_trajectory_joint_space(
        start_angles, end_angles, duration=2.0, dt=0.01
    )
    print(f"Trajectory generated: {len(angles)} points")
    print(f"Final angles (rad): {angles[-1]}")
    
    # VI. Control
    print("\nVI. CONTROL")
    print("-" * 60)
    
    # Low-level PID control
    pid_controllers = [robot.PIDController(kp=100, ki=10, kd=5, dt=0.01) 
                      for _ in range(6)]
    control_torques_pid = robot.compute_control_torque_pid(
        end_angles, start_angles, 
        np.zeros(6), np.zeros(6),
        pid_controllers
    )
    print(f"PID control torques (N*m): {control_torques_pid}")
    
    # High-level computed torque control
    kp = np.ones(6) * 100
    kd = np.ones(6) * 20
    control_torques_ct = robot.compute_control_torque_computed_torque(
        end_angles, start_angles,
        np.zeros(6), np.zeros(6),
        np.zeros(6), kp, kd
    )
    print(f"Computed torque control (N*m): {control_torques_ct}")
    
    # Visualization
    print("\nGenerating enhanced 3D visualizations...")
    
    # Enhanced single view
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    robot.plot_robot(joint_angles, ax1, show_frames=True)
    plt.savefig('robot_configuration_enhanced.png', dpi=150, bbox_inches='tight')
    print("Enhanced visualization saved to 'robot_configuration_enhanced.png'")
    plt.close()
    
    # Comparison view
    fig2 = plt.figure(figsize=(16, 6))
    ax2 = fig2.add_subplot(121, projection='3d')
    robot.plot_robot(joint_angles, ax2)
    ax2.set_title('Initial Configuration', fontsize=12, fontweight='bold')
    
    ax3 = fig2.add_subplot(122, projection='3d')
    robot.plot_robot(ik_solution, ax3)
    ax3.set_title('IK Solution Configuration', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('robot_configuration.png', dpi=150, bbox_inches='tight')
    print("Comparison visualization saved to 'robot_configuration.png'")
    plt.close()
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
