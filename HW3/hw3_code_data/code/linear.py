"""
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import time
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
import argparse
import matplotlib.pyplot as plt
from solvers import *
from utils import *

from PIL import Image as im
import pandas as pd


def create_linear_system(odoms, observations, sigma_odom, sigma_observation, n_poses, n_landmarks):
    """
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

    # Prepare Sigma^{-1/2}.
    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))

    H = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])
    # TODO: First fill in the prior to anchor the 1st pose at (0, 0)
    A[:2, :2] = sqrt_inv_odom

    # TODO: Then fill in odometry measurements
    X0 = np.zeros((1, 2), dtype=np.float32)
    for i in range(1, n_odom):
        x = 2 * i
        y = 2 * (i - 1)
        A[x : x + 2, y : y + 4] = sqrt_inv_odom @ H.copy()
        b[x : x + 2] = sqrt_inv_odom @ (odoms[i] - odoms[i - 1]).T

    # TODO: Then fill in landmark measurements
    pose_idx = observations[:, 1].astype("int64")
    land_idx = observations[:, 2].astype("int64")
    for p, l in zip(pose_idx, land_idx):
        i = (n_odom + 1 + l) * 2 - 1
        ip = 2 * p
        il = (n_poses + l) * 2 - 1
        # print(f"Pose {p} and Landmark {l} : ({i},{ip}) and ({i},{il})")
        A[i : i + 2, ip : ip + 2] = sqrt_inv_obs @ H[:, :2].copy()
        A[i : i + 2, il : il + 2] = sqrt_inv_obs @ H[:, 2:].copy()
        b[i : i + 2] = sqrt_inv_obs @ (observations[l, 2:] - odoms[p]).T

    plt.imshow(A, interpolation="nearest")
    plt.savefig("a_matrix.png")
    return csr_matrix(A), b


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="path to npz file")
    parser.add_argument(
        "--method", nargs="+", choices=["default", "pinv", "qr", "lu", "qr_colamd", "lu_colamd"], default=["default"], help="method"
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats in evaluation efficiency. Increase to ensure stablity.")
    args = parser.parse_args()

    data = np.load(args.data)
    print("Loaded Data!")
    # Plot gt trajectory and landmarks for a sanity check.
    gt_traj = data["gt_traj"]
    gt_landmarks = data["gt_landmarks"]
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], "b-", label="gt trajectory")
    plt.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c="b", marker="+", label="gt landmarks")
    plt.legend()
    plt.show()
    plt.savefig("init.png")

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odoms = data["odom"]
    observations = data["observations"]
    sigma_odom = data["sigma_odom"]
    sigma_landmark = data["sigma_landmark"]

    print(n_poses, n_landmarks)
    print(odoms.shape, observations.shape, sigma_odom.shape, sigma_landmark.shape)
    # print(odoms, observations, sigma_odom, sigma_landmark)

    # Build a linear system
    print("Building Linear System")
    A, b = create_linear_system(odoms, observations, sigma_odom, sigma_landmark, n_poses, n_landmarks)

    # Solve with the selected method
    for method in args.method:
        print(f"Applying {method}")

        total_time = 0
        total_iters = args.repeats
        for i in range(total_iters):
            start = time.time()
            x, R = solve(A, b, method)
            end = time.time()
            total_time += end - start
        print(f"{method} takes {total_time / total_iters}s on average")

        if R is not None:
            plt.spy(R)
            plt.show()

        traj, landmarks = devectorize_state(x, n_poses)

        # Visualize the final result
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)
