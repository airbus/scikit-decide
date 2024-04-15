import os
from abc import ABC, abstractmethod
from typing import Union

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

import skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.common_utils as Toolbox
import skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.interpolator.intergrid as intergrid
import skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.std_atm as std_atm
from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.common_utils import (
    convert,
)
from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.interpolator.WeatherInterpolator import (
    WeatherInterpolator,
)


class GenericInterpolator(ABC):
    @abstractmethod
    def interpol_field(self, X, **kwargs):
        """
        Interpol one field that is present in interpolators for array of 4d points

        :param X: array of points [time (in s), alt (in ft), lat, long]
        :param field: field of weather data to interpolate (could be 'temperature' or 'humidity'
        :return: array of interpolated values
        """
        ...

    @abstractmethod
    def render(self, ax, **kwargs):
        ...


def guess_axes(
    values: np.ndarray,
    lats: np.ndarray,
    longs: np.ndarray,
    levels: np.ndarray,
    times: np.ndarray,
):
    ndims = values.ndim
    shape = values.shape
    if ndims == 3:
        in_theory = ["times", "lats", "longs"]
    if ndims == 4:
        in_theory = ["times", "levels", "lats", "longs"]
    if ndims == 5:
        in_theory = ["times", "levels", "ensemble", "lats", "longs"]
    d = {"times": times, "levels": levels, "lats": lats, "longs": longs}
    l = {
        "times": len(times),
        "levels": len(levels),
        "lats": len(lats),
        "longs": len(longs),
    }
    for i in range(ndims):
        s = shape[i]
        if in_theory[i] != "ensemble":
            if s != len(d[in_theory[i]]):
                candidate = [t for t in l if l[t] == s]
                if len(candidate) > 0:
                    in_theory[i] = candidate[0]
                else:
                    in_theory[i] = "ensemble"
                    d["ensemble"] = range(s)
        else:
            d["ensemble"] = range(s)
    return in_theory, [d[x] for x in in_theory]


class GenericEnsembleInterpolator(GenericInterpolator):
    """
    Class used to store weather data, interpolate and plot weather forecast from .npz files
    """

    def __init__(
        self, file_npz, time_cut_index=None, fields=None, time_shift_s=0.0, order=1
    ):
        """
        Stores the weather data and build the interpolators on grid.
        """
        # Files Loading
        self.time_cut_index = time_cut_index
        self.axes = {}
        # self._auto_lock = Lock()
        self.time_shift_s = time_shift_s

        if isinstance(file_npz, (str, np.lib.npyio.NpzFile)):
            self.datas = (
                np.load(file_npz, allow_pickle=True)
                if isinstance(file_npz, str)
                else file_npz
            )
            if fields is None:
                fields = self.datas.keys()
            else:
                fields = [f for f in fields if f in list(self.datas.keys())]

            self.datas.allow_pickle = True
            items = {var: self.datas[var].item() for var in fields}
            self.lat_dict = {var: items[var]["lats"] for var in items}
            self.long_dict = {
                var: items[var]["longs"]
                if "longs" in items[var]
                else items[var]["lons"]
                for var in items
            }
            self.levels_dict = {
                var: items[var]["levels"] if "levels" in items[var] else [200.0]
                for var in items
            }
            self.time_dict = {
                var: items[var]["times"]
                if self.time_cut_index is None
                else items[var]["times"][
                    : min(self.time_cut_index, len(items[var]["times"]))
                ]
                for var in items
            }

            if time_shift_s != 0.0:
                for v in self.time_dict:
                    self.time_dict[v] += time_shift_s

            # items: U, V, T, R
            self.values = {var: items[var]["values"] for var in items}

            if self.time_cut_index is not None:
                index_cut = min(
                    self.time_cut_index,
                    len(self.time_dict[list(self.time_dict.keys())[0]]),
                )
                for var in self.values:
                    self.values[var] = self.values[var][:index_cut, :, :, :]
        elif isinstance(
            file_npz, dict
        ):  # Already loaded data in a dict (directly from parseWeather indeed)

            self.datas = file_npz
            if fields is None:
                fields = self.datas.keys()
            else:
                fields = [f for f in fields if f in list(self.datas.keys())]

            ### ??? ###
            # if fields is None:
            #     fields = self.datas.keys()
            # else:
            #     fields = [f for f in fields if f in list(self.datas.keys())]

            # self.datas.allow_pickle = True
            items = {var: self.datas[var] for var in fields}
            self.lat_dict = {var: items[var]["lats"] for var in fields}
            self.long_dict = {
                var: items[var]["longs"]
                if "longs" in items[var]
                else items[var]["lons"]
                for var in fields
            }
            self.levels_dict = {
                var: items[var]["levels"] if "levels" in items[var] else [200.0]
                for var in fields
            }

            self.time_dict = {var: items[var]["times"] for var in fields}
            if time_shift_s != 0.0:
                for v in self.time_dict:
                    self.time_dict[v] += time_shift_s
            self.values = {var: items[var]["values"] for var in fields}
            # for var in self.lat_dict:
            #     print(self.lat_dict[var].shape, "lats")
            #     print(self.long_dict[var].shape, "long")
            #     print(self.levels_dict[var].shape, "levels")
            #     print(self.time_dict[var].shape, "time")
            #     print(self.values[var].shape, "values")

        # one_field = list(self.values.keys())[0]
        for feat in self.lat_dict:
            if self.lat_dict[feat][-1] < self.lat_dict[feat][0]:
                self.lat_dict[feat] = self.lat_dict[feat][::-1]
                if self.values[feat].ndim == 4:
                    self.values[feat] = self.values[feat][:, :, ::-1, :]
                elif self.values[feat].ndim == 5:
                    self.values[feat] = self.values[feat][:, :, :, ::-1, :]
        self.interpol_dict = {}

        for var in self.values:
            # print(f'Building interpolator for {var}')
            # print(f'Levels dict: {self.levels_dict[var]}')
            self.levels_dict[var] = [111, 121]
            if (len(self.levels_dict[var]) == 0) or (len(self.levels_dict[var]) == 1):
                self.levels_dict[var] = np.array([30_000])

            axes = guess_axes(
                self.values[var],
                self.lat_dict[var],
                self.long_dict[var],
                self.levels_dict[var],
                self.time_dict[var],
            )

            self.axes[var] = axes
            self.interpol_dict[var] = intergrid.Intergrid(
                self.values[var],
                lo=[min(axes[1][i]) for i in range(len(axes[1]))],
                hi=[max(axes[1][i]) for i in range(len(axes[1]))],
                maps=axes[1],
                verbose=False,
                copy=True,
                order=order,
            )

    def add_new_field(self, origin_field, operation, axis, new_field):
        values = self.values[origin_field]
        new_values = operation(values, axis=axis)
        self.lat_dict[new_field] = self.lat_dict[origin_field]
        self.long_dict[new_field] = self.long_dict[origin_field]
        self.levels_dict[new_field] = self.levels_dict[origin_field]
        self.time_dict[new_field] = self.time_dict[origin_field]
        self.values[new_field] = new_values
        axes = guess_axes(
            self.values[new_field],
            self.lat_dict[new_field],
            self.long_dict[new_field],
            self.levels_dict[new_field],
            self.time_dict[new_field],
        )
        self.axes[new_field] = axes
        self.interpol_dict[new_field] = intergrid.Intergrid(
            self.values[new_field],
            lo=[min(axes[1][i]) for i in range(len(axes[1]))],
            hi=[max(axes[1][i]) for i in range(len(axes[1]))],
            maps=axes[1],
            verbose=False,
            copy=True,
            order=1,
        )

    def add_new_field_matrix(self, values, axes_values, new_field):
        self.values[new_field] = values
        self.axes[new_field] = axes_values
        self.interpol_dict[new_field] = intergrid.Intergrid(
            self.values[new_field],
            lo=[min(axes_values[i]) for i in range(len(axes_values))],
            hi=[max(axes_values[i]) for i in range(len(axes_values))],
            maps=axes_values,
            verbose=False,
            copy=True,
            order=1,
        )

    def interpol_field(self, X, field="CONVEXION"):
        """
        Interpol one field that is present in interpolators for array of 4d points

        :param X: array of points [time (in s), alt (in ft), lat, long], or [time (in s), alt (in ft), id-ensemble,
                                                                             lat, long]
        :param field: field of weather data to interpolate (could be 'temperature' or 'humidity'
        :return: array of interpolated values
        """
        # with self._auto_lock:
        return self.interpol_dict[field](X)

    def transform_long(self, long):
        """
        [Deprecated] should be replaced by modulo function...

        :param long: array of longitudes
        :return: array of longitude put in positive domain (modulo 360.)
        """
        return long
        # return np.where(long < 0, 360+long, long)

    def plot_field(
        self,
        field="issr",
        alt=35000.0,
        t: Union[float, np.ndarray] = 0.0,
        n_lat=180,
        n_long=720,
        ax=None,
    ):
        # plot the entire interpolate field
        # p = alt2press(alt)
        times = [t]
        n_time = 1
        if Toolbox.is_iterable(t):
            times = t
            n_time = len(t)
        down_long = min(self.long_dict[field])
        up_long = max(self.long_dict[field])
        down_lat = min(self.lat_dict[field])
        up_lat = max(self.lat_dict[field])
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(8, 10))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.stock_img()
            m = ax
        else:
            m = ax
        range_lat = np.linspace(down_lat, up_lat, n_lat)
        range_long = np.linspace(down_long, up_long, n_long)
        XX, YY = np.meshgrid(range_long, range_lat)
        x, y = XX, YY
        range_long = self.transform_long(range_long)
        if self.values[field].ndim == 4:
            values = np.array(
                [
                    [time, alt, range_lat[i], range_long[j]]
                    for i in range(n_lat)
                    for j in range(n_long)
                    for time in times
                ]
            )
        elif self.values[field].ndim == 3:
            values = np.array(
                [
                    [time, range_lat[i], range_long[j]]
                    for i in range(n_lat)
                    for j in range(n_long)
                    for time in times
                ]
            )
        Ut = np.resize(self.interpol_dict[field](values), (n_lat, n_long, n_time))
        i = 0
        cs = m.contourf(
            x,
            y,
            Ut[:, :, i],
            20,
            extent=[down_long, up_long, down_lat, up_lat],
            alpha=0.5,
            zorder=2,
        )
        plt.title("time " + str(times[i]))
        plt.draw()
        plt.pause(0.1)
        for i in range(1, n_time):
            for coll in cs.collections:
                plt.gca().collections.remove(coll)
            cs = m.contourf(
                x,
                y,
                Ut[:, :, i],
                20,
                extent=[down_long, up_long, down_lat, up_lat],
                alpha=0.5,
                zorder=2,
            )
            plt.title("time : " + str(times[i]))
            plt.draw()
            plt.pause(1)
        return m

    def plot_field_5d(
        self,
        field="CONVECTION",
        alt=35000.0,
        t: Union[float, np.ndarray] = 0.0,
        n_lat=180,
        index_forecast=0,
        n_long=720,
        ax=None,
        save=False,
        folder="",
        tag_file="weath",
    ):
        # plot the entire interpolate field
        # p = alt2press(alt)
        times = [t]
        n_time = 1
        if Toolbox.is_iterable(t):
            times = t
            n_time = len(t)
        down_long = min(self.long_dict[field])
        up_long = max(self.long_dict[field])
        down_lat = min(self.lat_dict[field])
        up_lat = max(self.lat_dict[field])
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(8, 10))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.stock_img()
            ax.set_xlim([down_long, up_long])
            ax.set_ylim([down_lat, up_lat])
            m = ax
        else:
            m = ax
        range_lat = np.linspace(down_lat, up_lat, n_lat)
        range_long = np.linspace(down_long, up_long, n_long)
        XX, YY = np.meshgrid(range_long, range_lat)
        x, y = XX, YY
        range_long = self.transform_long(range_long)
        values = np.array(
            [
                [time, alt, index_forecast, range_lat[i], range_long[j]]
                for i in range(n_lat)
                for j in range(n_long)
                for time in times
            ]
        )

        Ut = np.resize(self.interpol_dict[field](values), (n_lat, n_long, n_time))
        i = 0
        cs = ax.contour(
            x,
            y,
            Ut[:, :, i],
            extent=[down_long, up_long, down_lat, up_lat],
            alpha=0.9,
            zorder=2,
        )
        # cs = ax.imshow(Ut[:, :, i],
        #               extent=[down_long, up_long, down_lat, up_lat])
        plt.title("time " + str(times[i]))
        plt.draw()
        if save:
            j = 0
            t = "0" * (4 - len(str(j))) + str(j)
            plt.savefig(os.path.join(folder, t + "_" + tag_file + ".png"))
        plt.pause(0.1)
        for i in range(1, n_time):
            for coll in cs.collections:
                plt.gca().collections.remove(coll)
            ##cs.clear()
            # cs.set_data(Ut[:, :, i])
            cs = ax.contour(
                x,
                y,
                Ut[:, :, i],
                extent=[down_long, up_long, down_lat, up_lat],
                alpha=0.9,
                zorder=2,
            )
            plt.title("time : " + str(times[i]))
            plt.draw()
            if save:
                j += 1
                t = "0" * (4 - len(str(j))) + str(j)
                plt.savefig(os.path.join(folder, t + "_" + tag_file + ".png"))
            plt.pause(0.1)
        return m

    def render(self, ax, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(8, 10))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.stock_img()
        else:
            ax = ax
        keys = list(self.long_dict.keys())
        longs = self.long_dict[keys[0]]
        lats = self.lat_dict[keys[0]]
        dict_params = kwargs.get("kwargs", {})
        try:
            ax.imshow(
                self.values[dict_params["convexion_field"]][
                    0, 0, dict_params.get("index_forecast", 0), :, :
                ],
                cmap="hot",
                extent=[min(longs), max(longs), min(lats), max(lats)],
                interpolation="nearest",
                alpha=0.2,
            )
        except:
            pass


class GenericWindInterpolator(GenericEnsembleInterpolator, WeatherInterpolator):
    """
    Class used to store weather data, interpolate and plot weather forecast from .npz files
    """

    def __init__(
        self, file_npz, time_cut_index=None, fields=None, order_interp=1, time_shift_s=0
    ):
        """
        Stores the weather data and build the interpolators on grid.
        """
        super().__init__(
            file_npz, time_cut_index, fields=fields, time_shift_s=time_shift_s
        )
        if "U" not in self.values:
            ufield = "UGRD"
            vfield = "VGRD"
        else:
            ufield = "U"
            vfield = "V"
        self.norm_wind = np.sqrt(
            np.square(self.values[ufield]) + np.square(self.values[vfield])
        )
        self.angle_wind = np.arctan2(self.values[vfield], self.values[ufield])
        self.interpol_dict["norm-wind"] = intergrid.Intergrid(
            self.norm_wind,
            lo=[min(self.axes[ufield][1][i]) for i in range(len(self.axes[ufield][1]))],
            hi=[max(self.axes[ufield][1][i]) for i in range(len(self.axes[ufield][1]))],
            maps=self.axes[ufield][1],
            copy=True,
            verbose=False,
            order=1,
        )
        self.interpol_dict["argument-wind"] = intergrid.Intergrid(
            self.angle_wind,
            lo=[min(self.axes[ufield][1][i]) for i in range(len(self.axes[ufield][1]))],
            hi=[max(self.axes[ufield][1][i]) for i in range(len(self.axes[ufield][1]))],
            maps=self.axes[ufield][1],
            verbose=False,
            copy=True,
            order=1,
        )
        self.long_dict["norm-wind"] = self.long_dict[ufield]
        self.lat_dict["argument-wind"] = self.lat_dict[ufield]
        self.long_dict["argument-wind"] = self.long_dict[ufield]
        self.lat_dict["norm-wind"] = self.lat_dict[ufield]

    def transform_long(self, long):
        return long

    def interpol_wind_classic(self, lat, longi, alt=35000.0, t=0.0, index_forecast=0):
        # with self._auto_lock:
        p = std_atm.alt2press(alt, alt_units="ft", press_units="hpa")

        if len(self.axes["U"][1]) == 5:
            norm = self.interpol_dict["norm-wind"]([t, p, index_forecast, lat, longi])
            arg = self.interpol_dict["argument-wind"](
                [t, p, index_forecast, lat, longi]
            )

            result = norm * np.array([np.cos(arg), np.sin(arg)])
            return [norm, arg, result]

        if len(self.axes["U"][1]) == 4:
            norm = self.interpol_dict["norm-wind"]([t, p, lat, longi])
            arg = self.interpol_dict["argument-wind"]([t, p, lat, longi])

            result = norm * np.array([np.cos(arg), np.sin(arg)])
            return [norm, arg, result]

    def interpol_wind(self, X):
        """
        Interpol wind for an array of 4D points

        :param X: array of points [time (in s), alt (in ft), lat, long]
        :return: wind vector.
        """
        # with self._auto_lock:
        arg = self.interpol_dict["argument-wind"](X)
        return self.interpol_dict["norm-wind"](X) * np.array(
            [[np.cos(arg), np.sin(arg)]]
        )

    def plot_wind(
        self,
        alt=35000.0,
        down_long=-180.0,
        up_long=180.0,
        down_lat=-90.0,
        up_lat=90.0,
        t=0.0,
        n_lat=180,
        n_long=720,
        index_forecast=0,
        plot_wind=False,
        ax=None,
    ):
        """
        Plot the wind for a given coordinates window for a given altitude and for one time/or range of time

        :param alt: altitude couch to plot (in ft)
        :param down_long: min longitude
        :param up_long: max longitude
        :param down_lat: min latitude
        :param up_lat: max latitude
        :param t: value of time step (in second) or list/array of time step (in s)
        :type t: float or iterable
        :param n_lat: number of latitude discretized steps
        :param n_long: number of longitude discretized steps
        :param plot_wind: plot the vector field
        :param ax: Ax object where to plot the wind (possibily a precomputed basemap or classic ax object)
        """
        times = [t]
        n_time = 1
        if Toolbox.is_iterable(t):
            times = t
            n_time = len(t)
        if ax is None:
            fig, ax = plt.subplots(
                1, 1, figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()}
            )
            ax.stock_img()
        range_lat = np.linspace(down_lat, up_lat, n_lat)
        range_long = np.linspace(down_long, up_long, n_long)
        XX, YY = np.meshgrid(range_long, range_lat)
        x, y = XX, YY
        range_long = self.transform_long(range_long)
        values = np.array(
            [
                [time, alt, index_forecast, range_lat[i], range_long[j]]
                for i in range(n_lat)
                for j in range(n_long)
                for time in times
            ]
        )
        res = self.interpol_wind(values)
        Ut = np.resize(res[0, 0, :], (n_lat, n_long, n_time))
        Vt = np.resize(res[0, 1, :], (n_lat, n_long, n_time))
        Nt = np.sqrt(np.square(Ut) + np.square(Vt))
        # Nt = np.reshape(self.interpol_field(values, field='norm-wind'), (n_lat, n_long, n_time))

        i = 0
        CS = ax.contourf(x, y, Nt[:, :, i], 100, alpha=0.5, zorder=2)
        if plot_wind:
            if x.shape[0] > 100:
                q = ax.quiver(
                    x[::10, ::10],
                    y[::10, ::10],
                    Ut[::10, ::10, i],
                    Vt[::10, ::10, i],
                    alpha=0.8,
                )
            else:
                q = ax.quiver(x, y, Ut[:, :, i], Vt[:, :, i], alpha=0.8)
        plt.title("time : " + str(times[i]))
        for i in range(1, n_time):
            for coll in CS.collections:
                plt.gca().collections.remove(coll)
            CS = ax.contourf(x, y, Nt[:, :, i], 100, alpha=0.5, zorder=2)
            if plot_wind:
                q.remove()
                if x.shape[0] > 100:
                    q = ax.quiver(
                        x[::10, ::10],
                        y[::10, ::10],
                        Ut[::10, ::10, i],
                        Vt[::10, ::10, i],
                        alpha=0.8,
                    )
                else:
                    q = ax.quiver(x, y, Ut[:, :, i], Vt[:, :, i], alpha=0.8)
            plt.title("time : " + str(times[i]))
            plt.draw()
            plt.pause(0.1)
        return ax
