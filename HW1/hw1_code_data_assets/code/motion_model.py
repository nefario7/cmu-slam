"""
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import sys
import numpy as np
import math


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
        self._alpha1 = 0.001
        self._alpha2 = 0.001
        self._alpha3 = 0.001
        self._alpha4 = 0.001

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Update the function for vectorized 2D inputs
        """
        if (u_t1[0] == u_t0[0]) and (u_t1[1] == u_t0[1]) and (u_t1[2] == u_t0[2]):
            x_t1 = x_t0
            return x_t1

        x, y, theta = x_t0[:, 0], x_t0[:, 1], x_t0[:, 2]

        # * Motion Parameters
        d_rot_1 = np.arctan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        # d_trans = np.linalg.norm(u_t1[0:2] - u_t0[0:2])
        d_trans = np.sqrt((u_t1[0] - u_t0[0]) ** 2 + (u_t1[1] - u_t0[1]) ** 2)
        d_rot_2 = u_t1[2] - u_t0[2] - d_rot_1

        # * Relative Motion Parameters
        hd_rot_1 = d_rot_1 - self.sample_nd(self._alpha1 * np.power(d_rot_1, 2) + self._alpha2 * np.power(d_trans, 2))
        hd_trans = d_trans - self.sample_nd(
            self._alpha3 * np.power(d_trans, 2) + self._alpha4 * np.power(d_rot_1, 2) + self._alpha4 * np.power(d_rot_2, 2)
        )
        hd_rot_2 = d_rot_2 - self.sample_nd(self._alpha1 * np.power(d_rot_2, 2) + self._alpha2 * np.power(d_trans, 2))

        # * Contstrain [-pi, pi]
        hd_rot_1 = self.wrap_to_pi(hd_rot_1)
        hd_rot_2 = self.wrap_to_pi(hd_rot_2)

        # * Sample for state xt
        x_t1 = np.zeros_like(x_t0)
        x_t1[:, 0] = x + hd_trans * np.cos(theta + hd_rot_1)  # x
        x_t1[:, 1] = y + hd_trans * np.sin(theta + hd_rot_1)  # y
        x_t1[:, 2] = theta + hd_rot_1 + hd_rot_2  # theta

        return x_t1

    def prob_nd(self, a, bb):
        return np.power(2 * np.pi * bb, -0.5) * np.exp(-(a**2) / (2 * bb))

    def sample_nd(self, bb):
        return np.random.normal(loc=0, scale=np.sqrt(bb))

    def wrap_to_pi(self, angle):
        return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))
