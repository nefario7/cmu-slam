import numpy as np


def prob_nd(a, bb):
    return np.power(2 * np.pi * bb, -0.5) * np.exp(-(a**2) / (2 * bb))


def sample_nd(bb):
    return 0.5 * np.sum(np.random.normal(loc=0, scale=np.sqrt(bb), size=12))


def clip(a):
    mod = a % np.pi
    mod = mod - 2 * np.pi if mod > np.pi else mod
    return mod
