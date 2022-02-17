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
import matplotlib
from tqdm import tqdm

# matplotlib.use("TkAgg")
np.random.seed(15)


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap="Greys")
    plt.axis([0, 800, 0, 800])
    fig.show()


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c="r", marker=".")
    plt.savefig("{}/{:04d}.png".format(output_path, tstep))
    plt.pause(0.00001)
    scat.remove()


def init_particles_random(num_particles):

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

    """
    This version converges faster than init_particles_random
    """
    x0_vals = []
    y0_vals = []
    # * Iterating over till number of samples = number of particles needed
    while len(x0_vals) < num_particles:
        x = np.random.uniform(3000, 7500, (num_particles, 1))
        y = np.random.uniform(0, 7500, (num_particles, 1))
        y_idx = (y / 10.0).astype(int)
        x_idx = (x / 10.0).astype(int)
        for i in range(num_particles):
            # * Check if the location in occupancy map is free (P = 0)
            if np.abs(occupancy_map[y_idx[i], x_idx[i]]) == 0:
                if len(x0_vals) < num_particles:
                    x0_vals.append(x[i])
                    y0_vals.append(y[i])
        if len(x0_vals) == num_particles:
            break

    x0_vals = np.array(x0_vals)
    y0_vals = np.array(y0_vals)

    # occupancy_map = occupancy_map.T
    # freespace = occupancy_map * ((occupancy_map >= 0) & (occupancy_map <= 0.35))
    # freespace = np.where(occupancy_map == 0, 1, 0)
    # freespace_x = freespace.nonzero()[0] * 10
    # freespace_y = freespace.nonzero()[1] * 10
    # assert num_particles < freespace_x.shape[0], "Too many particles! No. of particles > Free space points"
    # x0_vals = np.random.choice(freespace_x, (num_particles, 1), replace=False)
    # y0_vals = np.random.choice(freespace_y, (num_particles, 1), replace=False)

    theta0_vals = np.random.uniform(-np.pi, np.pi, (num_particles, 1))
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles
    print(x0_vals.shape, y0_vals.shape, theta0_vals.shape, w0_vals.shape)
    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

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

    # * Initialize Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_map", default="hw1_code_data_assets/data/map/wean.dat")
    parser.add_argument("--path_to_log", default="hw1_code_data_assets/data/log/robotdata1.log")
    parser.add_argument("--output", default="hw1_code_data_assets/results")
    parser.add_argument("--num_particles", default=1000, type=int)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    os.chdir(r"D:\CMU\Academics\SLAM\Homeworks\HW1")

    # * Create folder to store frames
    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    logname = src_path_log.split("/")[-1]
    logname = logname.split(".")[0]
    print("Logfile = ", logname)
    save_path = args.output + "-" + logname
    os.makedirs(save_path, exist_ok=True)

    # * Object Definitions
    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, "r")

    # * Define the models for MCL
    motion_model = MotionModel()
    sensor_model = SensorModel(map_obj)
    resampler = Resampling()

    # * Particles and Initialization
    num_particles = args.num_particles
    # X_bar = init_particles_random(num_particles)
    X_bar = init_particles_freespace(num_particles, occupancy_map)

    # * Visualize MAP
    visualize_map(occupancy_map)

    # * Monte Carlo Localization Algorithm
    first_time_idx = True

    rand_idx = np.random.randint(0, num_particles)  # Random particle index for plotting rays

    start = time.time()
    for time_idx, line in enumerate(logfile):

        meas_type = line[0]
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=" ")

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        # if (time_stamp <= 0.0) | (meas_type == "O"):
        #     continue

        if meas_type == "L":
            odometry_laser = meas_vals[3:6]
            ranges = meas_vals[6:-1]

        if time_idx % 100 == 0:
            print("Processing time step {} at time {}s".format(time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # * Motion Model
        x_t0 = X_bar[:, 0:3]
        x_t1 = motion_model.update(u_t0, u_t1, x_t0)

        # * Sensor Model
        if meas_type == "L":
            z_t = ranges
            w_t, z_star = sensor_model.beam_range_finder_model(z_t, x_t1, odometry_laser)
            X_bar_new = np.hstack((x_t1, w_t))

            # * Plot the rays for a random particle
            # z_values = z_star[rand_idx]
            # x_val = X_bar[rand_idx, 0]
            # y_val = X_bar[rand_idx, 1]
            # theta_val = X_bar[rand_idx, 2]
            # print(z_values, x_val, y_val, theta_val)
            # angles = np.arange(np.pi / 2, -np.pi / 2, -np.pi / 36)
            # for i, a in enumerate(angles):
            #     p1 = [(x_val) / 10, (x_val + z_values[i] * np.cos(theta_val + a)) / 10]
            #     p2 = [(y_val) / 10, (y_val + z_values[i] * np.sin(theta_val + a)) / 10]
            #     (rays,) = plt.plot(p1, p2, "b", lw=0.5)
            # rays.remove()
        else:
            old_w_t = np.array(X_bar[:, 3])
            X_bar_new = np.hstack((x_t1, old_w_t[:, None]))

        X_bar = X_bar_new
        u_t0 = u_t1

        # * Resampling
        if meas_type == "L":
            X_bar = resampler.low_variance_sampler(X_bar)

        visualize_timestep(X_bar, time_idx, save_path)

end = time.time()
print(f"Time taken = {(end - start) / 60} mins\n")
