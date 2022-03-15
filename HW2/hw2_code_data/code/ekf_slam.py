"""
    Initially written by Ming Hsiao in MATLAB
    Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import numpy as np
import re
import matplotlib.pyplot as plt
import os

os.chdir(r"D:\CMU\Academics\SLAM\Homeworks\HW2\hw2_code_data\code")

np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)


def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array([3 * np.sqrt(a) * np.cos(phi[i]), 3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], "m")
    plt.draw()
    # plt.waitforbuttonpress(0)
    plt.pause(0.5)


def draw_traj_and_map(X, last_X, P, t, test_name):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], "b")
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c="b", linewidth=0.75)
    plt.plot(X[0], X[1], "*b")

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(X[3 + k * 2 : 3 + k * 2 + 2], P[3 + k * 2 : 3 + 2 * k + 2, 3 + 2 * k : 3 + 2 * k + 2], "r")
    else:
        for k in range(6):
            draw_cov_ellipse(X[3 + k * 2 : 3 + k * 2 + 2], P[3 + 2 * k : 3 + 2 * k + 2, 3 + 2 * k : 3 + 2 * k + 2], "g")

    plt.draw()
    plt.savefig("../plots/Trajectory_" + test_name + ".png")
    # plt.waitforbuttonpress(0)
    plt.pause(0.5)


def warp2pi(angle_rad):
    """
    TODO: warps an angle in [-pi, pi]. Used in the update step.

    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    if angle_rad > np.pi or angle_rad < -np.pi:
        angle_rad_warped = angle_rad - 2 * np.pi * np.floor((angle_rad + np.pi) / (2 * np.pi))
        return angle_rad_warped
    else:
        return angle_rad


def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    """
    TODO: initialize landmarks given the initial poses and measurements with their covariances
    \param init_measure Initial measurements in the form of (beta0, l0, beta1, l1, ...).
    \param init_measure_cov Initial covariance matrix of shape (2, 2) per landmark given parameters.
    \param init_pose Initial pose vector of shape (3, 1).
    \param init_pose_cov Initial pose covariance of shape (3, 3) given parameters.

    \return k Number of landmarks.
    \return landmarks Numpy array of shape (2k, 1) for the state.
    \return landmarks_cov Numpy array of shape (2k, 2k) for the uncertainty.
    """

    k = init_measure.shape[0] // 2

    landmark = np.zeros((2 * k, 1))
    landmark_cov = np.zeros((2 * k, 2 * k))

    x, y, theta = init_pose[0, 0], init_pose[1, 0], init_pose[2, 0]
    beta = init_measure[::2]
    r = init_measure[1::2]

    for i in range(k):
        p = 2 * i
        b = warp2pi(beta[i, 0] + theta)
        landmark[p] = x + r[i] * np.cos(warp2pi(theta + b))
        landmark[p + 1] = y + r[i] * np.sin(warp2pi(theta + b))

        # * Zero Cross covariances b/w bearing and range (Initialization Type 1)
        # landmark_cov[p : p + 2, p : p + 2] = init_measure_cov

        # * Non-zero cross covariances b/w bearing and range (Initialization Type 2)
        H_init = np.array([[1, 0, -r[i] * np.sin(b)], [0, 1, r[i] * np.cos(b)]], dtype=object)  # x, y, theta
        Q_init = np.array([[-r[i] * np.sin(b), np.cos(b)], [r[i] * np.cos(b), np.sin(b)]], dtype=object)  # beta, r
        landmark_cov[p : p + 2, p : p + 2] = H_init @ init_pose_cov @ H_init.T + Q_init @ init_measure_cov @ Q_init.T

    # print(landmark_cov)

    return k, landmark, landmark_cov


def predict(X, P, control, control_cov, k):
    """
    TODO: predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance of shape (3, 3) in the (x, y, theta) space given the parameters.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    """
    X_pre = np.zeros_like(X)
    P_pre = np.zeros_like(P)

    d, alpha = control[0], control[1]
    x, y, theta = X[:3]

    F = np.hstack((np.identity(3), np.zeros((3, 2 * k))))  # Mapping to higher dimension

    # * Predicted State - g(u, (x, y, theta)(t-1))
    g = np.array([d * np.cos(theta), d * np.sin(theta), alpha])  # Non-linear function

    X_pre = X + F.T @ g
    X_pre[2] = warp2pi(X_pre[2])

    # * Jacobian - G
    J = np.zeros((3, 3))
    J[0, 2] = -d * np.sin(theta)
    J[1, 2] = d * np.cos(theta)
    G = np.identity(3 + 2 * k) + F.T @ J @ F

    # control_cov_large = np.zeros_like(P)
    # control_cov_large[:3, :3] = control_cov

    # * Prediction
    # P_pre = G @ P @ G.T + control_cov_large  # Simple addition
    P_pre = G @ P @ G.T + F.T @ control_cov @ F  # Extra computation maybe

    # print("X_Pre = \n", X_pre)
    # print("P_Pre = \n", P_pre[:3, :3])
    # print(P[3:, 3:])
    return X_pre, P_pre


def update(X_pre, P_pre, measure, measure_cov, k):
    """
    TODO: update step in EKF SLAM with derived Jacobians.
    \param X_pre Predicted state vector of shape (3 + 2k, 1) from the predict step.
    \param P_pre Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    \param measure Measurement signal of shape (2k, 1).
    \param measure_cov Measurement covariance of shape (2, 2) per landmark given the parameters.
    \param k Number of landmarks.

    \return X Updated X state of shape (3 + 2k, 1).
    \return P Updated P covariance of shape (3 + 2k, 3 + 2k).
    """
    beta = measure[::2]
    r = measure[1::2]

    # x, y, theta = X_pre[0], X_pre[1], X_pre[2]
    # Landmarks in Global Frame
    # landmark = np.zeros((2 * k, 1))
    # for i in range(k):
    #     p = 2 * i
    #     landmark[p] = x + r[i] * np.cos(theta + beta[i])
    #     landmark[p + 1] = y + r[i] * np.sin(theta + beta[i])
    #     # print("Landmark", i, landmark[p], landmark[p + 1])
    # delta_x = landmark[::2] - x  # k x 1 Array
    # delta_y = landmark[1::2] - y  # k x 1 Array
    # print("Deltas : ", delta_x, delta_y)

    # * Updating Expected Pose and Covariance based on Landmarks
    for i in range(k):
        # print("Landmark ", i)
        x, y, theta = X_pre[0], X_pre[1], X_pre[2]

        lx = x + r[i] * np.cos(warp2pi(theta + beta[i]))
        ly = y + r[i] * np.sin(warp2pi(theta + beta[i]))

        dx = (lx - x)[0]
        dy = (ly - y)[0]

        # * Predicted and True Masurements for Landmark [i]
        measure_pre = np.zeros((2, 1))
        measure_pre[1] = np.sqrt(dx**2 + dy**2)
        measure_pre[0] = warp2pi(np.arctan2(dy, dx) - theta)  #! warp2pi?

        measure_tru = np.zeros((2, 1))
        p = 2 * i
        measure_tru[0] = measure[p]
        measure_tru[1] = measure[p + 1]
        # measure_tru = np.flip(measure_tru)

        # print("Measure Pre : ", measure_pre)
        # print("Measure Tru : ", measure_tru)

        # * Jacobians
        D = np.sqrt(dx**2 + dy**2)
        H_p = np.array([[-dx / D, -dy / D, 0], [dy / D**2, -dx / D**2, -1]])  # 2 x 3
        H_l = np.array([[dx / D, dy / D], [-dy / D**2, dx / D**2]])  # 2 x 2
        H_low = np.hstack((H_p, H_l))  # 2 x 5
        F = np.zeros((H_low.shape[1], 3 + 2 * k))

        # * Mapping Matrix
        F[:3, :3] = np.identity(3)
        F[3:, 2 * i + 3 : 2 * i + 5] = np.identity(2)
        H = H_low @ F

        # * Kalman Gain
        K = P_pre @ H.T @ np.linalg.inv(H @ P_pre @ H.T + measure_cov)
        X_pre = K @ (measure_tru - measure_pre) + X_pre
        P_pre = (np.identity(3 + 2 * k) - K @ H) @ P_pre

    # print("X_pre = \n", X_pre)
    # print("P_pre = \n", P_pre)
    return X_pre, P_pre


def evaluate(X, P, k, test_name):
    """
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    """
    # plt.waitforbuttonpress(0)
    print("Evaluation")
    # X, Y
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)
    plt.scatter(l_true[0::2], l_true[1::2])
    plt.draw()
    plt.savefig("../plots/Evaluation_" + test_name + ".png", dpi=1200)
    # plt.waitforbuttonpress(0)

    msum = 0
    for i in range(k):
        print(f"Landmark {i} :")
        idx = 2 * i
        pred = X[idx + 3 : idx + 5, 0]
        true = l_true[idx : idx + 2]
        diff = pred - true

        cova = P[idx + 3 : idx + 5, idx + 3 : idx + 5]

        maha = diff.T @ np.linalg.inv(cova) @ diff
        eucl = np.sqrt(diff @ diff.T)

        print(f"\tMahalanobis Distance = {maha} \tEuclidean Distance = {eucl}")
        msum += maha

    print("Average Maha = ", msum / k)
    plt.close()
    return None


def main(params, test_name=""):
    sig_x = params["sig_x"]
    sig_y = params["sig_y"]
    sig_alpha = params["sig_alpha"]
    sig_beta = params["sig_beta"]
    sig_r = params["sig_r"]

    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file and read the initial measurements
    data_file = open("../data/data.txt")
    line = data_file.readline()
    fields = re.split("[\t ]", line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = np.expand_dims(arr, axis=1)
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    pose = np.zeros((3, 1))
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    ##########
    # TODO: initialize landmarks
    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose, pose_cov)

    # Setup state vector X by stacking pose and landmark states
    # Setup covariance matrix P by expanding pose and landmark covariances
    X = np.vstack((pose, landmark))
    P = np.block([[pose_cov, np.zeros((3, 2 * k))], [np.zeros((2 * k, 3)), landmark_cov]])

    # Plot initial state and covariance
    last_X = X
    draw_traj_and_map(X, last_X, P, 0, test_name)

    # Core loop: sequentially process controls and measurements
    for line in data_file:
        fields = re.split("[\t ]", line)[:-1]
        arr = np.array([float(field) for field in fields])

        # Control
        if arr.shape[0] == 2:
            # print(f"{t}: Predict step")
            d, alpha = arr[0], arr[1]
            control = np.array([[d], [alpha]])

            ##########
            # TODO: predict step in EKF SLAM
            X_pre, P_pre = predict(X, P, control, control_cov, k)

            draw_traj_and_pred(X_pre, P_pre)

        # Measurement
        else:
            # print(f"{t}: Update step")
            measure = np.expand_dims(arr, axis=1)

            ##########
            # TODO: update step in EKF SLAM
            X, P = update(X_pre, P_pre, measure, measure_cov, k)

            draw_traj_and_map(X, last_X, P, t, test_name)
            last_X = X
            t += 1

    # EVAL: Plot ground truth landmarks and analyze distances
    evaluate(X, P, k, test_name)


if __name__ == "__main__":
    # TEST: Setup uncertainty parameters
    # * Reference Parameters\
    params = {"sig_x": 0.25, "sig_y": 0.1, "sig_alpha": 0.1, "sig_beta": 0.01, "sig_r": 0.08}

    # Default
    main(params)

    # Experiments
    # for sig, val in {"sig_x": 2.5, "sig_y": 1, "sig_alpha": 1, "sig_beta": 0.1, "sig_r": 0.8}.items():
    #     print("-" * 20, sig, "-" * 20)
    #     params = {"sig_x": 0.25, "sig_y": 0.1, "sig_alpha": 0.1, "sig_beta": 0.01, "sig_r": 0.08}
    #     params[sig] = val
    #     print(params)

    #     main(params, sig)
