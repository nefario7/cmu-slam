"""
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import sys
import numpy as np
import math
from utility import *


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """

    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.01
        self._alpha2 = 0.01
        self._alpha3 = 0.01
        self._alpha4 = 0.01

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        # Degree to Radians
        x_t0[2] = np.deg2rad(x_t0[2])
        u_t0[2] = np.deg2rad(u_t0[2])
        u_t1[2] = np.deg2rad(u_t1[2])

        x, y, theta = x_t0[0], x_t0[1], x_t0[2]

        # Motion Parameters
        d_rot_1 = np.arctan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        d_trans = np.linalg.norm(u_t1[0:2] - u_t0[0:2])
        d_rot_2 = u_t1[2] - u_t0[2] - d_rot_1

        # Relative Motion Parameters
        hd_rot_1 = d_rot_1 - sample_nd(self._alpha1 * np.power(d_rot_1, 2) + self._alpha2 * np.power(d_trans, 2))
        hd_trans = d_trans - sample_nd(
            self._alpha3 * np.power(d_trans, 2) + self._alpha4 * np.power(d_rot_1, 2) + self._alpha4 * np.power(d_rot_2, 2)
        )
        hd_rot_2 = d_rot_2 - sample_nd(self._alpha1 * np.power(d_rot_2, 2) + self._alpha2 * np.power(d_trans, 2))

        # Contstrain [-pi, pi]
        hd_rot_1 = clip(hd_rot_1)
        hd_rot_2 = clip(hd_rot_2)

        # Sample for state xt
        x_t1 = np.zeros_like(x_t0)
        x_t1[0] = x + hd_trans * np.cos(theta + hd_rot_1)  # x
        x_t1[1] = y + hd_trans * np.sin(theta + hd_rot_1)  # y
        x_t1[2] = theta + hd_rot_1 + hd_rot_2  # theta

        return x_t1
