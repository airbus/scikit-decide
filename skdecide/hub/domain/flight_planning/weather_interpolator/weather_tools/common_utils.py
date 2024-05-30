# -*- coding: utf-8 -*-
"""
Created on Tue May  3 12:09:06 2016 !!!
Useful custom functions callable by many part of the projects
@author: popo
"""
import collections
import math
import os

import matplotlib.pyplot as plt
import numpy as np

import skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.std_atm as std_atm


def tree():
    return collections.defaultdict(tree)


def my_extend(l1, l2):
    l = l1
    l.extend(l2)
    return l


def round_float(value, modulo=1000):
    n = int(round(float(value) / modulo))
    return n * modulo


def round_float_floor(value, modulo=1000):
    n = int(math.floor(float(value) / modulo))
    return n * modulo


def is_iterable(element):
    return isinstance(element, collections.Iterable)


def convert(val, init="kg/min", target="lb/h"):
    if init == target:
        return val
    if init == "kg/min" and target == "lb/h":
        lb = 2.20462
        return val * lb * 60
    if init == "lb" and target == "kg":
        lbtokg = 1.0 / 2.20462
        return lbtokg * val
    if init == "kg" and target == "lb":
        kgtolb = 2.20462
        return kgtolb * val
    if init == "ft" and target == "m":
        fttom = 0.3048
        return val * fttom
    if init == "m" and target == "ft":
        mtoft = 1.0 / 0.3048
        return val * mtoft
    if init == "ft" and target == "hPa":
        p0 = 1013.25
        t0 = 288.15
        alpha = 0.0065
        g0 = 9.80665
        r = 287.853
        p1 = 226.32
        t1 = 216.65
        return np.where(
            val > 36089,
            p1 * np.exp(-g0 * (convert(val, "ft", "m") - 11000.0) / (r * t1)),
            p0 * (1 - alpha / t0 * convert(val, "ft", "m")) ** (g0 / (alpha * r)),
        )
    if (init == "pa" or init == "Pa") and target == "ft":
        return std_atm.press2alt(val, press_units=init.lower(), alt_units=target)
    if init == "ft" and target == "Pa":
        return 10.0**2 * convert(val, init="ft", target="hPa")
    if init == "psia" and target == "Pa":
        return val * 6894.75728
    if init == "celsius" and target == "K":
        return val + 273.15
    if init == "K" and target == "celsius":
        return val - 273.15
    if init == "nm" and target == "km":
        return 1.852 * val
    if init == "km" and target == "nm":
        return val / 1.852
    if init == "m/s" and target == "kts":
        return 1.94384 * val
    if init == "kts" and target == "m/s":
        return val / 1.94384
    if init == "kl" and target == "kg":  # kl: kerosene liter
        return val * 0.8201
    if init == "kg" and target == "kl":
        return val / 0.8201


def get_regular_interval(l, n=100):
    """Return a regular array of length n of a sorted iterable

    :param l: A sorted iterable
    :param n: Number of discrete value we want
    :return: a sorted regular numpy array from l[0] to l[-1] with n step
    :rtype: `numpy.array`
    """
    return np.linspace(l[0], l[-1], n)


def intersect_interval(x, y):
    """X and Y given by [low_bound, high_bound]"""
    if len(x) < 2 or len(y) < 2 or x is None or y is None:
        return []
    lb = max(min(x), min(y))
    hb = min(max(x), max(y))
    return [lb, hb] if hb >= lb else []


def convert_str_tuple_to_tuple(str_tuple, conv=int):
    a = str_tuple.split(",")
    a[0] = conv(a[0][1:])
    a[1] = conv(a[1][1:-1])
    return a[0], a[1]


def convert_decimal_hour(hour):
    seconds = hour * 3600
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def get_absolute_path(filename, relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(filename), relative_path))


def get_absolute_path_from_rep(rep_name, relative_path):
    return os.path.abspath(os.path.join(rep_name, relative_path))


def monotonically_increasing(l):
    return all(x < y for x, y in zip(l, l[1:]))


def return_twin_axes(color_list=None, figsize=(10, 10)):
    if color_list is None:
        color_list = ["g", "r"]
    f, ax1 = plt.subplots(1, figsize=figsize)
    for tl in ax1.get_yticklabels():
        tl.set_color(color_list[0])
    ax2 = ax1.twinx()
    for tl in ax2.get_yticklabels():
        tl.set_color(color_list[1])
    return f, ax1, ax2
