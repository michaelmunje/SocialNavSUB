import cv2
import torch
import numpy as np
from typing import List, Tuple
import os
import pickle
import math

def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ],
    )
        
def localize_position_wrt_initial_pose(future_position, initial_position, initial_yaw):
    rotmat = yaw_rotmat(initial_yaw)
    return (future_position - initial_position).dot(rotmat)

class BEVPose:
    def __init__(self, x: float, y: float, yaw: float):
        self.x = x
        self.y = y
        self.yaw = yaw

    # minus
    def __sub__(self, other):
        delta_yaw = self.yaw - other.yaw
        if delta_yaw > np.pi:
            delta_yaw -= 2 * np.pi
        elif delta_yaw < -np.pi:
            delta_yaw += 2 * np.pi
        return BEVPose(self.x - other.x, self.y - other.y, delta_yaw)
    
    # plus
    def __add__(self, other):
        delta_yaw = self.yaw + other.yaw
        if delta_yaw > np.pi:
            delta_yaw -= 2 * np.pi
        elif delta_yaw < -np.pi:
            delta_yaw += 2 * np.pi
        return BEVPose(self.x + other.x, self.y + other.y, delta_yaw)

    def get_position_np(self):
        return np.array([self.x, self.y])
    
    def __repr__(self):
        return f"BEVPose(x={self.x}, y={self.y}, yaw={self.yaw})"

#  transform bev coordinates w.r.t. current pose to w.r.t initial pose
def get_bev_pose_wrt_initial_pose(bev_pose: BEVPose, bev_reference_pose: BEVPose, bev_target_pose: BEVPose) -> BEVPose:
    initial_yaw = bev_target_pose.yaw
    delta_bev_pose = bev_reference_pose - bev_target_pose
    delta_x, delta_y, delta_yaw = delta_bev_pose.x, delta_bev_pose.y, delta_bev_pose.yaw
    
    R = np.array([[np.cos(delta_yaw), -np.sin(delta_yaw)], 
                    [np.sin(delta_yaw), np.cos(delta_yaw)]])
    
    R2 = np.array([[np.cos(initial_yaw), np.sin(initial_yaw)], 
                    [-np.sin(initial_yaw), np.cos(initial_yaw)]])
    
    T = np.array([delta_x, delta_y])
    
    bev_position = bev_pose.get_position_np()
    bev_coords = R @ bev_position + (R2 @ T)
    transformed_yaw = bev_pose.yaw + delta_yaw if bev_pose.yaw is not None else None

    assert not np.isnan(bev_coords).any(), f'Nans in bev_coords: {bev_coords}'
    
    return BEVPose(bev_coords[0], bev_coords[1], transformed_yaw)

class Trajectory:
    def __init__(self, bev_poses: List[BEVPose], 
                 corresponding_timesteps: List[int], possible_timesteps: List[int], 
                 id: str, localize: bool, initial_yaw_estimation: bool = False):
        assert len(bev_poses) == len(corresponding_timesteps), 'Number of bev poses and corresponding timesteps do not match'
        assert len(corresponding_timesteps) <= len(possible_timesteps), f'Number of corresponding timesteps should be less than or equal to possible timesteps len({corresponding_timesteps}) > len({possible_timesteps})'
        self.bev_poses = bev_poses
        if localize:
            initial_position = self.bev_poses[0].get_position_np()
            initial_yaw = self.bev_poses[0].yaw
            initial_pose = BEVPose(initial_position[0], initial_position[1], initial_yaw)
            
            for i in range(len(self.bev_poses)):
                self.bev_poses[i].x, self.bev_poses[i].y = localize_position_wrt_initial_pose(
                    self.bev_poses[i].get_position_np(), 
                    initial_position, 
                    initial_yaw
                )
                self.bev_poses[i].yaw = self.bev_poses[i].yaw - initial_yaw
                self.bev_poses[i].yaw = self.bev_poses[i].yaw if self.bev_poses[i].yaw < np.pi else self.bev_poses[i].yaw - 2 * np.pi

        assert len(corresponding_timesteps) == len(self.bev_poses), 'Number of timesteps and bev poses do not match'
        self.corresponding_timesteps = corresponding_timesteps
        self.possible_timesteps = possible_timesteps
        self.id = id # can be robot, track id, coda id, etc.
        
        if initial_yaw_estimation:
            self.estimate_yaws()

    def get_timestep(self, idx: int):
        return self.corresponding_timesteps[idx]
    
    def get_timestep_idx(self, timestep: int):
        assert timestep in self.corresponding_timesteps, f'Timestep {timestep} not found in corresponding timesteps'
        return self.corresponding_timesteps.index(timestep)
    
    def get_pose_at_timestep(self, timestep: int):
        return self.bev_poses[self.get_timestep_idx(timestep)]
    
    def get_discontinuities(self) -> List[Tuple[int, int]]:
        # only care about them between first seen timestep and last seen timestep
        discontinuities: List[Tuple[int, int]] = []
        prev_timestep = self.corresponding_timesteps[0]
        for i in range(1, len(self.corresponding_timesteps)):
            next_timestep = self.corresponding_timesteps[i]
            prev_timestep_idx = self.possible_timesteps.index(prev_timestep)
            next_timestep_idx = self.possible_timesteps.index(next_timestep)
            if next_timestep_idx - prev_timestep_idx > 1:
                discontinuities.append((prev_timestep, next_timestep))
            prev_timestep = next_timestep
        return discontinuities
    
    def has_discontinuities(self) -> bool:
        return len(self.get_discontinuities()) > 0
    
    def interpolate_all_missing_poses(self):
        while self.has_discontinuities():
            discontinuities = self.get_discontinuities()
            # get first discontinuity
            prev_filled_timestep, next_filled_timestep = discontinuities[0]
            self.interpolate_missing_pose(prev_filled_timestep, next_filled_timestep)
    
    def interpolate_missing_pose(self, prev_timestep: int, next_timestep: int):
        prev_idx = self.get_timestep_idx(prev_timestep)
        target_idx = prev_idx + 1
        
        assert next_timestep - prev_timestep > 1, 'Two consecutive timesteps should have at least 1 frame gap'

        next_pose = self.get_pose_at_timestep(next_timestep)
        prev_pose = self.get_pose_at_timestep(prev_timestep)
        self.bev_poses.insert(target_idx, BEVPose(
            (next_pose.x + prev_pose.x) / 2,
            (next_pose.y + prev_pose.y) / 2,
            (next_pose.yaw + prev_pose.yaw) / 2
        ))
        # correct yaw to make sure valid range
        self.bev_poses[target_idx].yaw = self.bev_poses[target_idx].yaw if self.bev_poses[target_idx].yaw < np.pi else self.bev_poses[target_idx].yaw - 2 * np.pi
        
        # interpolate in the middle (or close to the middle) of the two timesteps
        # find middle timestep from possible timesteps
        next_timestep_idx = self.possible_timesteps.index(next_timestep)
        prev_timestep_idx = self.possible_timesteps.index(prev_timestep)
        new_timestep_idx = (next_timestep_idx + prev_timestep_idx) // 2
        new_timestep = self.possible_timesteps[new_timestep_idx]
        self.corresponding_timesteps.insert(target_idx, new_timestep)
            
    def kalman_smooth(self, speed_guess: float = 0.325, process_noise: List[float] = [1.0], measurement_noise: List[float] = [1.0]):
        if len(self.bev_poses) == 1:
            return
        # Initialize state vector: [x, y, vx, vy]
        n = len(self.bev_poses)
        # State [x, y, vx, vy] (positions and velocities)
        state = np.array([self.bev_poses[0].x, self.bev_poses[0].y, self.bev_poses[0].yaw, speed_guess])
        
        # State covariance matrix
        state_cov = np.eye(4)
        # Process noise
        if len(process_noise) == 1:
            Q = process_noise[0] * np.eye(4)
        elif len(process_noise) == 4:
            Q = np.diag(process_noise)
        else:
            raise ValueError("process_noise must be a list of either 1 or 4 elements")
        # Measurement matrix (we measure x and y positions only)
        H = np.array([[1, 0, 0, 0],  # we measure x
                        [0, 1, 0, 0]]) # we measure y
        
        # Measurement noise covariance matrix
        if len(measurement_noise) == 1:
            R = measurement_noise[0] * np.eye(2)
        elif len(measurement_noise) == 2:
            R = np.diag(measurement_noise)
        else:
            raise ValueError("process_noise must be a list of either 1 or 2 elements")
        # Identity matrix for updating
        I = np.eye(4)
        # Time step (assuming constant time step between poses)
        dt = 0.25  # You may need to adjust this based on your data
        # Forward pass (Kalman filter)
        forward_states = []
        forward_covs = []
        for i in range(n):
            # Get the current measurement (position)
            z = np.array([self.bev_poses[i].x, self.bev_poses[i].y])
            # Prediction step (non-linear)
            x, y, yaw, v = state
            state_pred = np.array([
                x + v * np.cos(yaw) * dt,
                y + v * np.sin(yaw) * dt,
                yaw,
                v
            ])
            # Jacobian of the state transition function
            F = np.array([
                [1, 0, -v * np.sin(yaw) * dt, np.cos(yaw) * dt],
                [0, 1,  v * np.cos(yaw) * dt, np.sin(yaw) * dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            # Covariance prediction
            state_cov = F @ state_cov @ F.T + Q
            # Kalman gain calculation
            S = H @ state_cov @ H.T + R  # residual covariance
            K = state_cov @ H.T @ np.linalg.inv(S)  # Kalman gain
            # Update step
            y = z - H @ state_pred  # measurement residual
            state = state_pred + K @ y  # state update
            state_cov = (I - K @ H) @ state_cov  # covariance update
            forward_states.append(state)
            forward_covs.append(state_cov)
        # Backward pass (RTS smoother)
        smoothed_states = [forward_states[-1]]
        smoothed_covs = [forward_covs[-1]]
        
        for i in range(n - 2, -1, -1):
            x, y, yaw, v = forward_states[i]
            F = np.array([
                [1, 0, -v * np.sin(yaw) * dt, np.cos(yaw) * dt],
                [0, 1,  v * np.cos(yaw) * dt, np.sin(yaw) * dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            P_pred = F @ forward_covs[i] @ F.T + Q
            C = forward_covs[i] @ F.T @ np.linalg.inv(P_pred)
            
            smoothed_state = forward_states[i] + C @ (smoothed_states[0] - F @ forward_states[i])
            smoothed_cov = forward_covs[i] + C @ (smoothed_covs[0] - P_pred) @ C.T
            
            smoothed_states.insert(0, smoothed_state)
            smoothed_covs.insert(0, smoothed_cov)
        self.bev_poses = [BEVPose(state[0], state[1], state[2]) for state in smoothed_states]
            
    def estimate_yaws(self):
        if len(self.bev_poses) == 1:
            self.bev_poses[0].yaw = None
            return

        # use the difference between consecutive poses to estimate the yaw
        for i in range(len(self.bev_poses) - 1):
            next_position = self.bev_poses[i + 1].get_position_np()
            current_position = self.bev_poses[i].get_position_np()
            displacement = next_position - current_position
            self.bev_poses[i].yaw = np.arctan2(displacement[1], displacement[0])
        self.bev_poses[-1].yaw = self.bev_poses[-2].yaw

    def __repr__(self):
        return f"Trajectory(poses={self.bev_poses})"
    
    def __len__(self):
        return len(self.bev_poses)

def get_camera_matrix(intrinsics: List[float]) -> torch.Tensor:
    fx, fy, cx, cy = intrinsics
    camera_matrix = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return camera_matrix

def get_inverse_camera_matrix(intrinsics: List[float]) -> torch.Tensor:
    camera_matrix = get_camera_matrix(intrinsics)
    inv_camera_matrix = torch.inverse(camera_matrix)
    return inv_camera_matrix

def quaternion_to_yaw(qw, qx, qy, qz):
    # Yaw calculation from quaternion
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return yaw

def transform_trajectory_to_initial_pose(traj: Trajectory, reference_frame_traj: Trajectory):
    initial_reference_pose = reference_frame_traj.bev_poses[0]
    for i in range(len(traj.bev_poses)):
        current_timestep = traj.corresponding_timesteps[i]
        reference_pose = reference_frame_traj.get_pose_at_timestep(current_timestep)
        traj.bev_poses[i] = get_bev_pose_wrt_initial_pose(
            traj.bev_poses[i], 
            reference_pose, 
            initial_reference_pose
        )

def test_trajectory_transforms():
    robot_bev_poses = [
            BEVPose(0, 0, 0),
            BEVPose(1, 0, 0),
            BEVPose(2, 0, 0),
            BEVPose(3, -1, -np.pi / 4),
            BEVPose(4, 2, np.pi / 2)
        ]
    timesteps = [0, 1, 2, 3, 4]
    robot_trajectory = Trajectory(
        bev_poses=robot_bev_poses,
        corresponding_timesteps=timesteps,
        possible_timesteps=timesteps,
        id='robot',
        localize=True,
        initial_yaw_estimation=False
    )
    # make sure bev poses are the same 
    same_bev_poses = True
    for i in range(len(robot_trajectory.bev_poses)):
        if robot_trajectory.bev_poses[i].x != robot_bev_poses[i].x or \
           robot_trajectory.bev_poses[i].y != robot_bev_poses[i].y or \
           robot_trajectory.bev_poses[i].yaw != robot_bev_poses[i].yaw:
            same_bev_poses = False
            break
    assert same_bev_poses, 'Bev poses are not the same'

    relative_poses = [
        BEVPose(0, 1, 0),
        BEVPose(0, 1, np.pi),
        BEVPose(-2, -2, -np.pi),
        BEVPose(5, 0, np.pi),
        BEVPose(1, -1, np.pi / 2)
    ]
    
    expected_adjusted_poses = [
        BEVPose(0, 1, 0),
        BEVPose(1, 1, np.pi),
        BEVPose(0, -2, -np.pi),
        BEVPose(3 + 5 * np.sin(np.pi / 4), -1 - 5 * np.cos(np.pi / 4), 3 * np.pi / 4),
        BEVPose(5, 3, np.pi)
    ]
    
    trajectory = Trajectory(
        bev_poses=relative_poses,
        corresponding_timesteps=timesteps,
        possible_timesteps=timesteps,
        id='robot',
        localize=False,
        initial_yaw_estimation=False
    )
    
    transform_trajectory_to_initial_pose(trajectory, robot_trajectory)
    
    for i in range(len(trajectory.bev_poses)):
        if trajectory.bev_poses[i].x != expected_adjusted_poses[i].x or \
           trajectory.bev_poses[i].y != expected_adjusted_poses[i].y or \
           trajectory.bev_poses[i].yaw != expected_adjusted_poses[i].yaw:
            same_bev_poses = False  
            break
    assert same_bev_poses, f'Bev poses are not the same at timestep {i}: {trajectory.bev_poses[i]} != {expected_adjusted_poses[i]}'
    
    # also make sure we can add offsets initial trajectory and it should be same since we localize
    original_robot_bev_poses = [BEVPose(pose.x, pose.y, pose.yaw) for pose in robot_trajectory.bev_poses]
    
    offset_x = 5
    offset_y = 3
    new_bev_poses = []
    for i in range(len(original_robot_bev_poses)):
        new_x = original_robot_bev_poses[i].x + offset_x
        new_y = original_robot_bev_poses[i].y + offset_y
        new_yaw = original_robot_bev_poses[i].yaw
        new_bev_poses.append(BEVPose(
            new_x,
            new_y,
            new_yaw
        ))
    new_trajectory = Trajectory(
        bev_poses=new_bev_poses,
        corresponding_timesteps=timesteps,
        possible_timesteps=timesteps,
        id='robot',
        localize=True,
        initial_yaw_estimation=False
    )

    same_bev_poses = True
    for i in range(len(new_trajectory.bev_poses)):
        if new_trajectory.bev_poses[i].x != original_robot_bev_poses[i].x or \
           new_trajectory.bev_poses[i].y != original_robot_bev_poses[i].y or \
           new_trajectory.bev_poses[i].yaw != original_robot_bev_poses[i].yaw:
            same_bev_poses = False
            break
    assert same_bev_poses, f'Bev poses are not the same at timestep {i}: {new_trajectory.bev_poses[i]} != {original_robot_bev_poses[i]}'

# this class will contain the trajectory, the corresponding timesteps, the possible timesteps, the id of the object, and the 2d bounding box
class TrackedObject:
    def __init__(self, trajectory: Trajectory, 
                 corresponding_timesteps: List[int], 
                 possible_timesteps: List[int], 
                 id: str, 
                 position_differences: List[float],
                 corresponding_bboxes: List[List[int]]):
        self.trajectory = trajectory
        self.corresponding_timesteps = corresponding_timesteps
        self.possible_timesteps = possible_timesteps
        self.id = id
        self.corresponding_bboxes = corresponding_bboxes
        self.position_differences = position_differences
        self.closest_x = min([pose.x for pose in self.trajectory.bev_poses])
        self.closest_y = min([pose.y for pose in self.trajectory.bev_poses])
        self.furthest_x = max([pose.x for pose in self.trajectory.bev_poses])
        self.furthest_y = max([pose.y for pose in self.trajectory.bev_poses])
        self.n_timesteps_appeared = len(self.trajectory.bev_poses)
        
        self.color: Tuple[int, int, int] = None
        self.label: int = None
    
    def assign_color(self, color: Tuple[int, int, int]):
        self.color = color
        
    def assign_label(self, label: int):
        self.label = label
    