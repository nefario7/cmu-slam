"""
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 1
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 5

        # Laser
        self.n_beams = 30

        # Occupancy Map
        self.occupancy_map = occupancy_map
        self.map_x = occupancy_map.get_map_size_x()
        self.map_y = occupancy_map.get_map_size_y()

        # Robot Intrinsic
        self.gap = 25  # cm

    def wrap_to_pi(self, a):
        mod = a % np.pi
        mod = mod - 2 * np.pi if mod > np.pi else mod
        return mod

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        # prob_zt1 = 1.0
        prb_zt1 = 0  # Log Likelihood
        z_samples = [z_t1_arr[i] for i in range(0, 180, self._subsampling)]
        z_star = self.__ray_cast(x_t1, z_samples)

        for i in range(z_samples):
            prob = self.__get_prob(z_star[i], z_samples[i])  # 4 x 1
            z = self.__learn_intrinsic_parameters(prob)  # 1 x 4
            p = z @ p
            # prob_zt1 = prob_zt1 * p
            prob_zt1 += np.log(p)

        prob_zt1 = np.exp(prob_zt1)
        return prob_zt1

    def __learn_intrinsic_parameters(self, prob):
        # ? Update the parameters/ weights for each probability distributions

        return np.array([[self.z_hit], [self.z_short], [self.z_max], [self.z_rand]])

    def __ray_cast(self, x_t1, z_samples):
        # ? How to calculate z_star
        x = x_t1[0]
        y = x_t1[1]
        #! Check if degtorad or vice versa needed
        theta = x_t1[2]

        #! Test what fucking works
        theta_l = self.theta - np.pi / 2
        x_l = x + self.gap * np.cos(theta_l)
        y_l = y + self.gap * np.sin(theta_l)

        laser_step = np.pi * 2 / self.n_beams
        z_star = np.zeros(n_beams)
        for beam in range(self.n_beams):
            pass

        return z_star
