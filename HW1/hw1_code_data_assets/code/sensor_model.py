"""
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import numpy as np
import numpy.ma as ma
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader
from tqdm import tqdm


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, map_obj):
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
        self._min_probability = 0.7  #!0.35

        # Used in sampling angles in ray casting
        self._subsampling = 20  #! 5
        self.n_beams = int(180 / self._subsampling)
        self.max_laser_range = 1000
        self.step_size = 250  #! 10

        assert self._max_range == self.max_laser_range

        # Occupancy Map
        self.occupancy_map = map_obj.get_map()
        self.map_x = map_obj.get_map_size_x()
        self.map_y = map_obj.get_map_size_y()
        self.res = map_obj._resolution

        # Robot Intrinsic
        self.gap = 25  # cm

    def __wrap_to_pi(self, a):
        mod = a % np.pi
        mod = mod - 2 * np.pi if mod > np.pi else mod
        return mod

    def __get_prob(self, z_star, z_samples):
        mask_hit_rand = ma.masked_inside(z_samples, 0, self._max_range).mask
        mask_short = (z_samples < z_star) & (z_samples >= 0)
        mask_max = z_samples == self._max_range

        # * Hit
        p_hit = (np.exp(-1 / 2 * (z_samples - z_star) ** 2 / (self._sigma_hit**2))) / (np.sqrt(2 * np.pi * self._sigma_hit**2))
        p_hit = p_hit * mask_hit_rand
        # print(p_hit)

        # * Short
        eta = 1.0 / (1 - np.exp(-self._lambda_short * z_star))
        p_short = eta * self._lambda_short * np.exp(-self._lambda_short * z_samples)
        p_short = p_short * mask_short
        # print(p_short)

        # * Max
        p_max = np.ones_like(z_samples)
        p_max = p_max * mask_max
        # print(p_max)

        # * Rand
        p_rand = np.ones_like(z_samples) / self._max_range
        p_rand = p_rand * mask_hit_rand
        # print(p_rand)

        return np.vstack([p_hit, p_short, p_max, p_rand])

    def __learn_intrinsic_parameters(self, prob):
        # ? Update the parameters/ weights for each probability distributions
        # TODO

        return np.array([[self.z_hit], [self.z_short], [self.z_max], [self.z_rand]])

    def __ray_cast(self, x_t1):
        # ? How to calculate z_star
        m = x_t1.shape[0]
        z_star = np.zeros((m, self.n_beams))

        x = x_t1[:, 0]
        y = x_t1[:, 1]
        theta = x_t1[:, 2]

        # Correction for Laser Offset
        theta_l = theta - (np.pi / 2)
        x_l = x + self.gap * np.cos(theta_l)
        y_l = y + self.gap * np.sin(theta_l)

        d_arr = np.arange(1, self.max_laser_range / self.step_size + 1) * self.step_size
        print(d_arr)
        d_mat = np.tile(d_arr, (m, 1))

        x_l = np.tile(x_l, (d_arr.shape[0], 1)).T
        y_l = np.tile(y_l, (d_arr.shape[0], 1)).T

        angle_step = np.pi * 2 / self.n_beams
        angle = -np.pi / 6  # -np.pi / 2
        for ray in tqdm(range(self.n_beams), desc="Ray Casting"):
            # print(f"\n___________________ Beam - {ray} ___________________\n")
            # * Projecting steps on X and Y axes
            x_s = x_l + d_mat * np.cos(angle)
            y_s = y_l + d_mat * np.sin(angle)
            # print(x_s)

            # * Clipping distances if beyond map
            x_s = np.clip(x_s, 0, self.map_x)
            y_s = np.clip(y_s, 0, self.map_y)
            # print("XS clipped")
            # print(x_s)

            # * Getting indexes from distances (0 - 799 that's why -1)
            x_s_idx = np.rint(x_s / self.res - 1).astype(int)
            y_s_idx = np.rint(y_s / self.res - 1).astype(int)
            # print("XS indexes")
            # print(x_s_idx)

            # * Probabilites from Occupancy Map
            # probs = self.occupancy_map[x_s_idx, y_s_idx]
            probs = np.random.rand(m, d_arr.shape[0])
            # print("P")
            # print(probs)

            # * Keep probabilites > Threshold and get index of first 'True'
            mask = probs > self._min_probability
            indexes = np.where(mask.any(axis=1), mask.argmax(axis=1), -1)
            # print(indexes)

            # * Update Z-star
            z_star[:, ray] = d_arr[indexes]
            # print("Temp Z")
            # print(z_star)

            angle += angle_step

        return z_star

    def beam_range_finder_model(self, z_t1, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame] [M particles x 3]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        prob_zt1 = np.zeros((x_t1.shape[0], 1))

        z_samples = z_t1[:: self._subsampling].copy()  # (180 / sub-sample) x 1
        z_star = self.__ray_cast(x_t1)  # M x (180 / sub-sample)

        for p in range(x_t1.shape[0]):
            prob = self.__get_prob(z_star[p], z_samples)
            prob = prob.T

            # TODO
            z = self.__learn_intrinsic_parameters(prob)
            p_total = p @ z
            prob_zt1 += np.log(p)

        prob_zt1 = np.exp(prob_zt1)
        return prob_zt1
