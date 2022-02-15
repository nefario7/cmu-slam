"""
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """

    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled = np.zeros_like(X_bar)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled = list()
        X_bar_weights = X_bar[:, 3].copy()
        M = X_bar.shape[0]

        X_bar_weights = X_bar_weights / np.sum(X_bar_weights)
        if np.sum(X_bar_weights) == 0:
            print("ENCOUNTERED ZERO DIVISION!")
        r = np.random.uniform(0, 1.0 / M)

        c = X_bar_weights[0]
        i = 0

        for m in range(1, M + 1):
            u = r + (m - 1) / M
            while u > c:
                i = i + 1
                c = c + X_bar_weights[i]

            X_bar_resampled.append(X_bar[i])
            # np.set_printoptions(precision=4)
            # print(f"{i} -> {X_bar[i]}")

        X_bar_resampled = np.array(X_bar_resampled)
        return X_bar_resampled
