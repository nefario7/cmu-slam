"""
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import argparse
import matplotlib.pyplot as plt
from solvers import *
from utils import *

from tqdm import tqdm


def warp2pi(angle_rad):
    """
    Warps an angle in [-pi, pi]. Used in the update step.
    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad = angle_rad - 2 * np.pi * np.floor((angle_rad + np.pi) / (2 * np.pi))
    return angle_rad


def init_states(odoms, observations, n_poses, n_landmarks):
    """
    Initialize the state vector given odometry and observations.
    """
    traj = np.zeros((n_poses, 2))
    landmarks = np.zeros((n_landmarks, 2))
    landmarks_mask = np.zeros((n_landmarks)).astype("bool")

    for i in range(len(odoms)):
        traj[i + 1, :] = traj[i, :] + odoms[i, :]

    for i in range(len(observations)):
        pose_idx = int(observations[i, 0])
        landmark_idx = int(observations[i, 1])

        if not landmarks_mask[landmark_idx]:
            landmarks_mask[landmark_idx] = True

            pose = traj[pose_idx, :]
            theta, d = observations[i, 2:]

            landmarks[landmark_idx, 0] = pose[0] + d * np.cos(theta)
            landmarks[landmark_idx, 1] = pose[1] + d * np.sin(theta)

    return traj, landmarks


def odometry_estimation(x, i):
    """
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from (odometry between pose i and i+1)
    \return odom Odometry (\Delta x, \Delta) in the shape (2, )
    """
    # TODO: return odometry estimation
    idx = 2 * i
    odom = np.array([x[idx + 2] - x[idx], x[idx + 3] - x[idx + 1]])

    return odom


def bearing_range_estimation(x, i, j, n_poses):
    """
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return obs Observation from pose i to landmark j (theta, d) in the shape (2, )
    """
    # TODO: return bearing range estimations
    dy = x[2 * n_poses + 2 * j + 1] - x[2 * i + 1]
    dx = x[2 * n_poses + 2 * j] - x[2 * i]
    # print("dy", land[2 * j + 1], pose[2 * i + 1])

    obs = np.array([warp2pi(np.arctan2(dy, dx)), np.sqrt(dy**2 + dx**2)])
    return obs


def compute_meas_obs_jacobian(x, i, j, n_poses):
    """
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return jacobian Derived Jacobian matrix in the shape (2, 4)
    """
    # TODO: return jacobian matrix
    dy = x[2 * n_poses + 2 * j + 1] - x[2 * i + 1]
    dx = x[2 * n_poses + 2 * j] - x[2 * i]

    D1 = np.sqrt(dy**2 + dx**2)
    D2 = np.power(D1, 2)

    Hl = np.array([[dy / D2, -dx / D2, -dy / D2, dx / D2], [-dx / D1, -dy / D1, dx / D1, dy / D1]])

    return Hl


def create_linear_system(x, odoms, observations, sigma_odom, sigma_observation, n_poses, n_landmarks):
    """
    \param x State vector x at which we linearize the system.
    \param odoms Odometry measurements between i and i+1 in the global coordinate system. Shape: (n_odom, 2).
    \param observations Landmark measurements between pose i and landmark j in the global coordinate system. Shape: (n_obs, 4).
    \param sigma_odom Shared covariance matrix of odometry measurements. Shape: (2, 2).
    \param sigma_observation Shared covariance matrix of landmark measurements. Shape: (2, 2).

    \return A (M, N) Jacobian matrix.
    \return b (M, ) Residual vector.
    where M = (n_odom + 1) * 2 + n_obs * 2, total rows of measurements.
          N = n_poses * 2 + n_landmarks * 2, length of the state vector.
    """

    n_odom = len(odoms)
    n_obs = len(observations)

    M = (n_odom + 1) * 2 + n_obs * 2
    N = n_poses * 2 + n_landmarks * 2

    A = np.zeros((M, N))
    b = np.zeros((M,))

    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))

    # TODO: First fill in the prior to anchor the 1st pose at (0, 0)
    A[:2, :2] = np.eye(2)

    # TODO: Then fill in odometry measurements
    Hp = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])
    for i in range(n_odom):
        idxx = 2 * (i + 1)
        idxy = 2 * i

        e_odom = odoms[i] - odometry_estimation(x, i)
        A[idxx : idxx + 2, idxy : idxy + 4] = sqrt_inv_odom @ Hp
        b[idxx : idxx + 2] = sqrt_inv_odom @ e_odom

    # TODO: Then fill in landmark measurements
    for i in range(n_obs):
        pose_idx = int(observations[i, 0])
        land_idx = int(observations[i, 1])

        idx = 2 * (n_odom + i + 1)
        idxl = 2 * (n_poses + land_idx)
        idxp = 2 * pose_idx

        Hl = compute_meas_obs_jacobian(x, pose_idx, land_idx, n_poses)

        estimate = bearing_range_estimation(x, pose_idx, land_idx, n_poses)
        measured = observations[i, 2:]
        e_meas = np.array([warp2pi(measured[0] - estimate[0]), (measured[1] - estimate[1])])

        A[idx : idx + 2, idxp : idxp + 2] = sqrt_inv_obs @ Hl[:, :2]
        A[idx : idx + 2, idxl : idxl + 2] = sqrt_inv_obs @ Hl[:, 2:]
        b[idx : idx + 2] = sqrt_inv_obs @ e_meas

    return csr_matrix(A), b


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", default="../data/2d_nonlinear.npz")
    parser.add_argument(
        "--method",
        nargs="+",
        choices=["default", "pinv", "qr", "lu", "qr_colamd", "lu_colamd", "custom_lu"],
        default=["default"],
        help="method",
    )

    args = parser.parse_args()
    print("\n" + "-" * 80)
    print("Loading Data")
    data = np.load(args.data)

    # Plot gt trajectory and landmarks for a sanity check.
    gt_traj = data["gt_traj"]
    gt_landmarks = data["gt_landmarks"]
    # plt.plot(gt_traj[:, 0], gt_traj[:, 1], "b-")
    # plt.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c="b", marker="+")
    # plt.show()

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odom = data["odom"]
    observations = data["observations"]
    sigma_odom = data["sigma_odom"]
    sigma_landmark = data["sigma_landmark"]

    # print(n_poses, n_landmarks)
    # print(odom.shape, observations.shape, sigma_odom.shape, sigma_landmark.shape)
    # print(odom[:3], observations[:3], sigma_odom[:3], sigma_landmark[:3])

    # Initialize: non-linear optimization requires a good init.
    for method in args.method:
        print(f"Applying {method}")
        traj, landmarks = init_states(odom, observations, n_poses, n_landmarks)

        print("Before Optimization")
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks, method, optim=False)

        # Iterative optimization
        x = vectorize_state(traj, landmarks)
        for i in tqdm(range(10), desc="Optimization"):
            A, b = create_linear_system(x, odom, observations, sigma_odom, sigma_landmark, n_poses, n_landmarks)
            dx, _ = solve(A, b, method)
            if method in ["qr", "qr_colamd"]:
                dx = dx.reshape((dx.shape[0],))
            x = x + dx

        traj, landmarks = devectorize_state(x, n_poses)

        print("After Optimization")
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks, method, optim=True)
