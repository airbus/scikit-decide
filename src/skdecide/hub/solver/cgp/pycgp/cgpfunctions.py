# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np


def f_sum(args):
    return 0.5 * (args[0] + args[1])


def f_aminus(args):
    return 0.5 * (abs(args[0] - args[1]))


def f_mult(args):
    return args[0] * args[1]


def f_exp(args):
    return (np.exp(args[0]) - 1.0) / (np.exp(1.0) - 1.0)


def f_abs(args):
    return abs(args[0])


def f_sqrt(args):
    return np.sqrt(abs(args[0]))


def f_sqrtxy(args):
    return np.sqrt(args[0] * args[0] + args[1] * args[1]) / np.sqrt(2.0)


def f_squared(args):
    return args[0] * args[0]


def f_pow(args):
    return pow(abs(args[0]), abs(args[1]))


def f_one(args):
    return 1.0


def f_zero(args):
    return 0.0


def f_inv(args):
    if args[0] != 0.0:
        return args[0] / abs(args[0])
    else:
        return 0.0


def f_gt(args):
    return float(args[0] > args[1])


def f_acos(args):
    return math.acos(args[0]) / np.pi


def f_asin(args):
    return 2.0 * math.asin(args[0]) / np.pi


def f_atan(args):
    return 4.0 * math.atan(args[0]) / np.pi


def f_min(args):
    return np.min(args)


def f_max(args):
    return np.max(args)


def f_round(args):
    return round(args[0])


def f_floor(args):
    return math.floor(args[0])


def f_ceil(args):
    return math.ceil(args[0])
