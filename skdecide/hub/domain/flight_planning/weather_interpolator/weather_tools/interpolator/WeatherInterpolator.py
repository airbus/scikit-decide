from abc import ABC, abstractmethod

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

import skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.common_utils as Toolbox
import skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.interpolator.intergrid as intergrid


class WeatherInterpolator(ABC):
    @abstractmethod
    def interpol_wind(self, X):
        """
        Interpol wind for an array of 4D points

        :param X: array of points [time (in s), alt (in ft), lat, long]
        :return: wind vector.
        """
        ...

    @abstractmethod
    def interpol_field(self, X, field="temperature"):
        """
        Interpol one field that is present in interpolators for array of 4d points

        :param X: array of points [time (in s), alt (in ft), lat, long]
        :param field: field of weather data to interpolate (could be 'temperature' or 'humidity'
        :return: array of interpolated values
        """
        ...

    @abstractmethod
    def interpol_wind_classic(self, lat, longi, alt, **kwargs):
        """
        Interpolate the wind.
        :param lat: latitude in degree
        :param longi: longitude in degree
        :param alt: altitude in feet
        :param kwargs: other parameters for the given weather data underlying. example : id of the ensemble forecast.
        :return:
        """
        ...

    @abstractmethod
    def render(self, ax, **kwargs):
        ...


class WeatherForecastInterpolator(WeatherInterpolator):
    """
    Class used to store weather data, interpolate and plot weather forecast from .npz files
    """

    def __init__(self, file_npz, time_cut_index=None, order_interp=1):
        """
        Stores the weather data and build the interpolators on grid.
        """
        # Files Loading
        self.time_cut_index = time_cut_index
        if isinstance(file_npz, (str, np.lib.npyio.NpzFile)):
            self.datas = np.load(file_npz) if isinstance(file_npz, str) else file_npz

            self.lat_dict = {
                var: self.datas[var].item()["lats"] for var in self.datas.keys()
            }
            self.long_dict = {
                var: self.datas[var].item()["longs"] for var in self.datas.keys()
            }
            self.alt_dict = {
                var: self.datas[var].item()["levels"] for var in self.datas.keys()
            }
            self.time_dict = {
                var: self.datas[var].item()["times"]
                if self.time_cut_index is None
                else self.datas[var].item()["times"][
                    : min(self.time_cut_index, len(self.datas[var].item()["times"]))
                ]
                for var in self.datas.keys()
            }
            # Data Extraction
            # self.u_wind = self.datas["U"].item()["values"]
            # self.v_wind = self.datas["V"].item()["values"]
            self.humidity = self.datas["R"].item()["values"]
            self.temperature = self.datas["T"].item()["values"]

            if self.time_cut_index is not None:
                index_cut = min(
                    self.time_cut_index,
                    len(self.time_dict[list(self.time_dict.keys())[0]]),
                )
                # self.u_wind = self.u_wind[:index_cut, :, :, :]
                # self.v_wind = self.v_wind[:index_cut, :, :, :]
                self.humidity = self.humidity[:index_cut, :, :, :]
                self.temperature = self.temperature[:index_cut, :, :, :]
        elif isinstance(
            file_npz, dict
        ):  # Already loaded data in a dict (directly from parseWeather indeed)

            self.datas = file_npz
            self.lat_dict = {var: self.datas[var]["lats"] for var in self.datas.keys()}
            self.long_dict = {
                var: self.datas[var]["longs"] for var in self.datas.keys()
            }
            self.alt_dict = {
                var: self.datas[var]["levels"] for var in self.datas.keys()
            }
            self.time_dict = {
                var: self.datas[var]["times"] for var in self.datas.keys()
            }
            # Data Extraction
            # self.u_wind = self.datas["U"]["values"]
            # self.v_wind = self.datas["V"]["values"]
            self.humidity = self.datas["R"]["values"]
            self.temperature = self.datas["T"]["values"]

        for feat in self.lat_dict:
            if self.lat_dict[feat][-1] < self.lat_dict[feat][0]:
                self.lat_dict[feat] = self.lat_dict[feat][::-1]
                # if feat == "U":
                #     self.u_wind = self.u_wind[:, :, ::-1, :]
                # elif feat == "V":
                #     self.v_wind = self.v_wind[:, :, ::-1, :]
                if feat == "R":
                    self.humidity = self.humidity[:, :, ::-1, :]
                elif feat == "T":
                    self.temperature = self.temperature[:, :, ::-1, :]

        # self.norm_wind = np.sqrt(np.square(self.u_wind) + np.square(self.v_wind))
        # self.angle_wind = np.arctan2(self.v_wind, self.u_wind)

        self.interpol_dict = {}
        # self.interpol_dict["wind"] = {
        #     "norm": intergrid.Intergrid(
        #         self.norm_wind,
        #         lo=[
        #             min(self.time_dict["U"]),
        #             min(self.alt_dict["U"]),
        #             min(self.lat_dict["U"]),
        #             min(self.long_dict["U"]),
        #         ],
        #         hi=[
        #             max(self.time_dict["U"]),
        #             max(self.alt_dict["U"]),
        #             max(self.lat_dict["U"]),
        #             max(self.long_dict["U"]),
        #         ],
        #         maps=[
        #             self.time_dict["U"],
        #             self.alt_dict["U"],
        #             self.lat_dict["U"],
        #             self.long_dict["U"],
        #         ],
        #         verbose=False,
        #         order=order_interp,
        #     ),

        #     "argument": intergrid.Intergrid(
        #         self.angle_wind,
        #         lo=[
        #             min(self.time_dict["U"]),
        #             min(self.alt_dict["U"]),
        #             min(self.lat_dict["U"]),
        #             min(self.long_dict["U"]),
        #         ],
        #         hi=[
        #             max(self.time_dict["U"]),
        #             max(self.alt_dict["U"]),
        #             max(self.lat_dict["U"]),
        #             max(self.long_dict["U"]),
        #         ],
        #         maps=[
        #             self.time_dict["U"],
        #             self.alt_dict["U"],
        #             self.lat_dict["U"],
        #             self.long_dict["U"],
        #         ],
        #         verbose=False,
        #         order=order_interp,
        #     ),
        # }
        self.interpol_dict["humidity"] = intergrid.Intergrid(
            self.humidity,
            lo=[
                min(self.time_dict["R"]),
                min(self.alt_dict["R"]),
                min(self.lat_dict["R"]),
                min(self.long_dict["R"]),
            ],
            hi=[
                max(self.time_dict["R"]),
                max(self.alt_dict["R"]),
                max(self.lat_dict["R"]),
                max(self.long_dict["R"]),
            ],
            maps=[
                self.time_dict["R"],
                self.alt_dict["R"],
                self.lat_dict["R"],
                self.long_dict["R"],
            ],
            verbose=False,
            order=order_interp,
        )
        self.interpol_dict["temperature"] = intergrid.Intergrid(
            self.temperature,
            lo=[
                min(self.time_dict["T"]),
                min(self.alt_dict["T"]),
                min(self.lat_dict["T"]),
                min(self.long_dict["T"]),
            ],
            hi=[
                max(self.time_dict["T"]),
                max(self.alt_dict["T"]),
                max(self.lat_dict["T"]),
                max(self.long_dict["T"]),
            ],
            maps=[
                self.time_dict["T"],
                self.alt_dict["T"],
                self.lat_dict["T"],
                self.long_dict["T"],
            ],
            verbose=False,
            order=order_interp,
        )

    def interpol_wind_classic(self, lat, longi, alt=35000.0, t=0.0, **kwargs):
        """
        Interpol the wind in one 4D point

        :param lat: latitude
        :param longi: longitude
        :param alt: altitude (in ft)
        :param t: time in second
        :return: [wind strength, direction, wind wector]
        """
        pass
        # if longi < 0:
        #     longi += 360.0
        # norm = self.interpol_dict["wind"]["norm"]([t, alt, lat, longi])
        # arg = self.interpol_dict["wind"]["argument"]([t, alt, lat, longi])
        # result = norm * np.array([np.cos(arg), np.sin(arg)])
        # return [norm, arg, result]

    def interpol_wind(self, X):
        """
        Interpol wind for an array of 4D points

        :param X: array of points [time (in s), alt (in ft), lat, long]
        :return: wind vector.
        """
        pass
        # arg = self.interpol_dict["wind"]["argument"](X)
        # return self.interpol_dict["wind"]["norm"](X) * np.array(
        #     [[np.cos(arg), np.sin(arg)]]
        # )

    def interpol_field(self, X, field="temperature"):
        """
        Interpol one field that is present in interpolators for array of 4d points

        :param X: array of points [time (in s), alt (in ft), lat, long]
        :param field: field of weather data to interpolate (could be 'temperature' or 'humidity'
        :return: array of interpolated values
        """
        return self.interpol_dict[field](X)

    def transform_long(self, long):
        """
        [Deprecated] should be replaced by modulo function...

        :param long: array of longitudes
        :return: array of longitude put in positive domain (modulo 360.)
        """
        return np.where(long < 0, 360 + long, long)

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
                [time, alt, range_lat[i], range_long[j]]
                for i in range(n_lat)
                for j in range(n_long)
                for time in times
            ]
        )
        res = self.interpol_wind(values)
        Ut = np.resize(res[0, 0, :], (n_lat, n_long, n_time))
        Vt = np.resize(res[0, 1, :], (n_lat, n_long, n_time))
        Nt = np.sqrt(np.square(Ut) + np.square(Vt))
        # Nt = self.interpol_field(values, field="norw-wind")

        i = 0
        CS = ax.contourf(x, y, Nt[:, :, i], 20, alpha=0.5, zorder=2)
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
            CS = ax.contourf(x, y, Nt[:, :, i], 20, alpha=0.5, zorder=2)
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

    def plot_wind_noised(
        self,
        alt=35000.0,
        down_long=-180.0,
        up_long=180.0,
        down_lat=-90.0,
        up_lat=90.0,
        t=0.0,
        n_lat=180,
        n_long=720,
        plot_wind=False,
        mean_noised_norm=1.05,
        scale_noised_norm=0.01,
        mean_noised_arg=0.1,
        scale_noised_arg=0.01,
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
                [time, alt, range_lat[i], range_long[j]]
                for i in range(n_lat)
                for j in range(n_long)
                for time in times
            ]
        )

        arg = self.interpol_dict["wind"]["argument"](values)
        norm = self.interpol_dict["wind"]["norm"](values)
        noised_norm = np.random.normal(
            mean_noised_norm, scale=scale_noised_norm, size=norm.shape
        )
        norm = noised_norm * norm
        noised_arg = np.random.normal(
            mean_noised_arg, scale=scale_noised_arg, size=arg.shape
        )

        res_noised = norm * np.array(
            [[np.cos(arg + noised_arg), np.sin(arg + noised_arg)]]
        )
        res = self.interpol_wind(values)
        Ut_noised = np.resize(res_noised[0, 0, :], (n_lat, n_long, n_time))
        Vt_noised = np.resize(res_noised[0, 1, :], (n_lat, n_long, n_time))
        Nt_noised = np.sqrt(np.square(Ut_noised) + np.square(Vt_noised))
        Ut = np.resize(res[0, 0, :], (n_lat, n_long, n_time))
        Vt = np.resize(res[0, 1, :], (n_lat, n_long, n_time))
        Nt = np.sqrt(np.square(Ut - Ut_noised) + np.square(Vt - Vt_noised))
        i = 0
        # CS = ax.contourf(x, y, Nt[:, :, i], 20, alpha=0.5, zorder=2)
        if plot_wind:
            if x.shape[0] > 100:
                q = ax.quiver(
                    x[::10, ::10],
                    y[::10, ::10],
                    Ut_noised[::10, ::10, i],
                    Vt_noised[::10, ::10, i],
                    alpha=0.8,
                )
            else:
                q = ax.quiver(
                    x,
                    y,
                    Ut_noised[:, :, i] - Ut[:, :, i],
                    Vt_noised[:, :, i] - Vt[:, :, i],
                    alpha=0.8,
                )
        plt.title("time : " + str(times[i]))
        for i in range(1, n_time):
            for coll in CS.collections:
                plt.gca().collections.remove(coll)
            CS = ax.contourf(x, y, Nt_noised[:, :, i], 20, alpha=0.5, zorder=2)
            if plot_wind:
                q.remove()
                if x.shape[0] > 100:
                    q = ax.quiver(
                        x[::10, ::10],
                        y[::10, ::10],
                        Ut_noised[::10, ::10, i],
                        Vt_noised[::10, ::10, i],
                        alpha=0.8,
                    )
                else:
                    q = ax.quiver(
                        x, y, Ut_noised[:, :, i], Vt_noised[:, :, i], alpha=0.8
                    )
            plt.title("time : " + str(times[i]))
            plt.draw()
            plt.pause(0.1)
        return ax

    def plot_matrix_wind(self, index_alt=10, index_time=0, ax=None):
        """
        [Deprecated]

        Plot the wind matrix directly (no interpolation contrary
        to :func:`BEN3_G.Contrails.WeatherForecastInterpolator.plot_wind`
        """
        down_long = -180.0
        up_long = 180.0
        down_lat = -90.0
        up_lat = 90.0
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(8, 10))
            ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0))
            ax.stock_img()
            m = ax
        else:
            m = ax
        range_lat = np.linspace(down_lat, up_lat, 1000)
        range_long = np.linspace(down_long, up_long, 1000)
        XX, YY = np.meshgrid(range_long, range_lat)
        x, y = XX, YY
        norm = np.sqrt(
            np.square(self.temperature[index_time, index_alt, :, :])
            + np.square(self.temperature[index_time, index_alt, :, :])
        )
        ax.imshow(
            norm,
            cmap="hot",
            interpolation="nearest",
            extent=[-180.0, 180.0, -90.0, 90.0],
            alpha=0.5,
        )
        # plt.gca().invert_yaxis()
        # ax.colorbar()

    def plot_matrix_wind_noised(self, index_alt=10, index_time=0, ax=None):
        """
        [Deprecated]

        Plot the wind matrix directly (no interpolation contrary
        to :func:`BEN3_G.Contrails.WeatherForecastInterpolator.plot_wind`
        """
        down_long = -180.0
        up_long = 180.0
        down_lat = -90.0
        up_lat = 90.0
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(8, 10))
            ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0))
            ax.stock_img()
            m = ax
        else:
            m = ax
        range_lat = np.linspace(down_lat, up_lat, 1000)
        range_long = np.linspace(down_long, up_long, 1000)
        XX, YY = np.meshgrid(range_long, range_lat)
        x, y = XX, YY
        norm = np.sqrt(
            np.square(self.u_wind[index_time, index_alt, ::-1, :])
            + np.square(self.v_wind[index_time, index_alt, ::-1, :])
        )
        ax.imshow(
            norm,
            cmap="hot",
            interpolation="nearest",
            extent=[-180.0, 180.0, 90.0, -90.0],
            alpha=0.5,
        )
        plt.gca().invert_yaxis()
        ax.colorbar()

    def plot_field(
        self, field="issr", alt=35000.0, t=0.0, n_lat=180, n_long=720, ax=None
    ):
        """
        Plot a field for a given altitude and for one time/or range of time

        :param alt: altitude couch to plot (in ft)
        :param t: value of time step (in second) or list/array of time step (in s)
        :param n_lat: number of latitude discretized steps
        :param n_long: number of longitude discretized steps
        :param ax: Ax object where to plot the wind (possibily a precomputed basemap or classic ax object)
        """
        # plot the entire interpolate field
        # p = alt2press(alt)
        p = alt
        times = [t]
        n_time = 1
        if Toolbox.is_iterable(t):
            times = t
            n_time = len(t)

        down_long = -180.0
        up_long = 180.0
        down_lat = -90.0
        up_lat = 90.0
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
        Ut = np.zeros((n_lat, n_long))
        values = np.array(
            [
                [time, alt, range_lat[i], range_long[j]]
                for i in range(0, n_lat)
                for j in range(0, n_long)
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
        plt.title("time " + str(times[i]))
        plt.draw()
        plt.pause(0.1)
        for i in range(1, n_time):
            for coll in cs.collections:
                plt.gca().collections.remove(coll)
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
            plt.pause(1)
        return m

    def render(self, ax, **kwargs):
        pass
        # self.plot_matrix_wind(index_alt=0, index_time=0, ax=ax)
