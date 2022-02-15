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

# from map_reader import MapReader
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
        self._z_hit = 100
        self._z_short = 10
        self._z_max = 10
        self._z_rand = 1000
        self._z_params = np.array([self._z_hit, self._z_short, self._z_max, self._z_rand])

        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35  #!0.7

        # Used in sampling angles in ray casting
        self._subsampling = 1  #! 20
        self.n_beams = int(180 / self._subsampling)
        self.max_laser_range = 1000
        self.step_size = 5  #! 250

        assert self._max_range == self.max_laser_range

        # Occupancy Map
        self.occupancy_map = map_obj.get_map().T
        self.map_x = map_obj.get_map_size_x()
        self.map_y = map_obj.get_map_size_y()
        self.res = map_obj._resolution

        # Robot Intrinsic
        self.gap = 25  # cm

    def __get_prob(self, z_star, z_samples):
        mask_hit = (z_samples <= self._max_range) & (z_samples >= 0)
        mask_short = (z_samples < z_star) & (z_samples >= 0)
        mask_max = z_samples == self._max_range
        mask_rand = (z_samples < self._max_range) & (z_samples >= 0)

        # * Hit
        p_hit = (np.exp((-1 / 2) * ((z_samples - z_star) ** 2) / (self._sigma_hit**2))) / (np.sqrt(2 * np.pi * self._sigma_hit**2))
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

    def __learn_intrinsic_parameters(self, prob, z_star, z_samples):
        # ? Whether to use subsampled - z_samples / full - z_t1
        norm = np.sum(prob, axis=0)
        epsilon = 0.0000001
        norm = np.power(norm + epsilon, -1)

        exp = prob @ np.diagflat(norm)
        mod_z = z_samples.shape[0]

        self._z_hit = mod_z * np.sum(exp[0])
        self._z_short = mod_z * np.sum(exp[1])
        self._z_max = mod_z * np.sum(exp[2])
        self._z_rand = mod_z * np.sum(exp[3])

        self._sigma_hit = np.sqrt(np.sum(exp[0] * np.power(z_samples - z_star, 2)) / np.sum(exp[0]))
        self._lambda_short = np.sum(exp[1]) / np.sum(exp[1] * z_samples)

        return np.array([self._z_hit, self._z_short, self._z_max, self._z_rand])

    def __ray_cast(self, x_t1):
        # ? How to calculate z_star
        m = x_t1.shape[0]
        z_star = np.zeros((m, self.n_beams))

        x = x_t1[:, 0]
        y = x_t1[:, 1]
        theta = x_t1[:, 2]

        #! Correction for Laser Offset
        theta_l = theta - (np.pi / 2)
        x_l = x + self.gap * np.cos(theta_l)
        y_l = y + self.gap * np.sin(theta_l)

        d_arr = np.arange(1, self.max_laser_range / self.step_size + 1) * self.step_size
        d_mat = np.tile(d_arr, (m, 1))

        x_l = np.tile(x_l, (d_arr.shape[0], 1)).T
        y_l = np.tile(y_l, (d_arr.shape[0], 1)).T

        angles = np.arange(np.pi / 2, -np.pi / 2, -np.pi / self.n_beams)
        for ray, angle in enumerate(angles):
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
            probs = self.occupancy_map[x_s_idx, y_s_idx]
            # probs = np.random.rand(m, d_arr.shape[0])
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

        return z_star

    def beam_range_finder_model(self, z_t1, x_t1, i):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame] [M particles x 3]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        prob_zt1 = np.zeros((x_t1.shape[0], 1))

        z_samples = z_t1[:: self._subsampling].copy()  # (180 / sub-sample) x 1
        z_star = self.__ray_cast(x_t1)  # M x (180 / sub-sample)

        # for p in tqdm(range(x_t1.shape[0]), desc=f"Beam Range Finder {i}"):
        for p in range(x_t1.shape[0]):
            prob_dist = self.__get_prob(z_star[p], z_samples)
            # z_params = self.__learn_intrinsic_parameters(prob_dist, z_star[p], z_samples)
            z_params = self._z_params
            prob_total = np.log(z_params @ prob_dist)

            # prob_zt1[p] = np.prod(prob_total)
            prob_zt1[p] = np.exp(np.sum(prob_total))

        # print(prob_zt1)
        return prob_zt1, z_star
