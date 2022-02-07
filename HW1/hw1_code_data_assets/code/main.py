"""
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time
from tqdm import tqdm


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap="Greys")
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c="r", marker="o")
    plt.savefig("{}/{:04d}.png".format(output_path, tstep))
    plt.pause(0.00001)
    scat.remove()


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    TODO : Add your code here
    This version converges faster than init_particles_random
    """
    X_bar_init = np.zeros((num_particles, 4))

    return X_bar_init


if __name__ == "__main__":
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_map", default="hw1_code_data_assets/data/map/wean.dat")
    parser.add_argument("--path_to_log", default="hw1_code_data_assets/data/log/robotdata1.log")
    parser.add_argument("--output", default="results")
    parser.add_argument("--num_particles", default=500, type=int)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    os.chdir(r"D:\CMU\Academics\SLAM\Homeworks\HW1")

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, "r")

    # * Define the models for MCL
    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    # * Particles and Initialization
    num_particles = args.num_particles
    X_bar = init_particles_random(num_particles, occupancy_map)
    # X_bar = init_particles_freespace(num_particles, occupancy_map)

    # * Monte Carlo Localization Algorithm : Main Loop
    if args.visualize:
        visualize_map(occupancy_map)

    first_time_idx = True
    for time_idx, line in tqdm(enumerate(logfile), desc="Working on Log File"):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        """
        L -94.234001 -139.953995 -1.342158 -88.567719 -164.303391 -1.342158
        66 66 66 66 66 65 66 66 65 66 66 66 66 66 67 67 67 66 67 66 67 67 67 68 68 68 69 67
        530 514 506 508 494 481 470 458 445 420 410 402 393 386 379 371 365 363 363 364 358
        353 349 344 339 335 332 328 324 321 304 299 298 294 291 288 287 284 282 281 277 277
        276 274 273 271 269 268 267 266 265 265 264 263 263 263 262 261 261 261 261 261 193
        190 189 189 192 262 262 264 194 191 190 190 193 269 271 272 274 275 277 279 279 281
        283 285 288 289 292 295 298 300 303 306 309 314 318 321 325 329 335 340 360 366 372
        378 384 92 92 91 89 88 87 86 85 84 83 82 82 81 81 80 79 78 78 77 76 76 76 75 75 74
        74 73 73 72 72 72 71 72 71 71 71 71 71 71 71 71 70 70 70 70 0.025466
        """
        # L : laser scan measurement, O : odometry measurement

        meas_type = line[0]
        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=" ")
        print(meas_vals.shape)
        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        #! ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        if meas_type == "L":
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        # if time_idx % 100 == 0:
        #     print("Processing time step {} at time {}s".format(time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        #! Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        #! Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        for m in range(0, num_particles):
            """
            * MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)

            """
            * SENSOR MODEL
            """
            if meas_type == "L":
                z_t = ranges
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
                X_bar_new[m, :] = np.hstack((x_t1, w_t))
            # else:
            #     X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        X_bar = X_bar_new
        u_t0 = u_t1

        """
        * RESAMPLING
        """
        X_bar = resampler.low_variance_sampler(X_bar)

        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output)

        # #! Vectorization
        # x_t0 = X_bar[:, 0:3]  # M x 3
        # x_t1 = motion_model.update(u_t0, u_t1, x_t0)  # M x 3
        # if meas_type == "L":
        #     z_t = ranges  # 1 x 180
        #     w_t = sensor_mode.beam_range_finder_model(z_t, x_t1)  # ? M x 1
        #     X_bar_new[m, :] = np.hstack((x_t1, w_t))  # ? M x 4

        # X_bar = X_bar_new  # Update particles data
        # u_t0 = u_t1  # Update the previous state to current state

        # X_bar = resampler.low_variance_sampler(X_bar)  # ? M x 4
