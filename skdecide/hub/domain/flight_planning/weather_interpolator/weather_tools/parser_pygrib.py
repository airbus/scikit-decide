import datetime
import os
import sys

import numpy as np
import pygrib
import pytz


def computeTimeStamps(dates, times, steps):
    """
    This method computes the time stamp from dates, times and step from forecast.

    :param dates: List of dates with the following format YYYYMMDD.
    :type dates: list
    :param times: List of forecast time provided as integers.
    :type times: list
    :param steps: List of forecast steps provided as integers.
    :type steps: list
    :return: List of timestamps as datetimes.
    :rtype: list
    """
    timestamps = set()
    for date in dates:
        date_string = "{0:-08d}".format(date)
        for time in times:
            hour = int(time / 100)
            date_object = datetime.datetime(
                year=int(date_string[0:4]),
                month=int(date_string[4:6]),
                day=int(date_string[6:8]),
                hour=hour,
                minute=0,
                second=0,
                tzinfo=pytz.utc,
            )
            for step in steps:
                forecast_datetime = date_object + datetime.timedelta(hours=int(step))
                # forecast_datetime = forecast_datetime.replace(tzinfo=pytz.utc)
                # forecast_datetime.timestamp()
                timestamps.add(forecast_datetime.timestamp())
    return list(timestamps)


def computeTimeStamp(date, time, step):
    """
    This method computes a single time stamp value from a forecast date, time and step.

    :param date: date with the following format YYYYMMDD.
    :type date: int
    :param time: forecast time value.
    :type time: int
    :param step: forecast step value.
    :type step: int
    :return: timestamp
    :rtype: datetime.datetime
    """
    date_string = "{0:-08d}".format(date)
    hour = int(time / 100)
    date_object = datetime.datetime(
        year=int(date_string[0:4]),
        month=int(date_string[4:6]),
        day=int(date_string[6:8]),
        hour=hour,
        minute=0,
        second=0,
        tzinfo=pytz.utc,
    )
    forecast_datetime = date_object + datetime.timedelta(hours=int(step))
    # forecast_datetime.replace(tzinfo=pytz.utc)
    return datetime.datetime.timestamp(forecast_datetime)


def flip_matrix(matrix):
    """
    Method to flip reverse the order of the elements in the latitude axis of an array.

    :param matrix: Dictionary with the matrix to perform the flip.
    :type matrix: dict
    :return: Dictionary with the flipped matrix.
    :rtype: dict
    """
    for var in matrix:
        if matrix[var]["lats"][-1] < matrix[var]["lats"][0]:
            matrix[var]["lats"] = matrix[var]["lats"][::-1]
            matrix[var]["values"] = np.flip(
                matrix[var]["values"], matrix[var]["values"].ndim - 2
            )


class GribPygribUniqueForecast(object):
    """
    This class transforms a grib file into a dictionary containing the values of desired parameters. It allows the
    extraction of just one variable or set of variables corresponding to: CAT, Windshear, Icing, Wind uncertainty and
    convection. It is based on Pygrib library.

    """

    def __init__(
        self, grib_path, grib_name, selected_forecast_dates=None, selected_levels=None
    ):
        """
        Initialization.

        :param grib_path: Grib file path.
        :type grib_path: str
        :param grib_name: Grib file name.
        :type grib_name: str
        :param selected_forecast_dates: List of selected forecast dates.
        :type selected_forecast_dates: list
        :param selected_levels: List of selected levels.
        :type selected_levels: list
        """

        # open grib file with pygrib
        gribs = pygrib.open(os.path.join(grib_path, grib_name))
        self.gribs = gribs

        # save latitudes and longitudes
        latlons = gribs[1].latlons()
        latitudes = latlons[0][:, 0]
        longitudes = latlons[1][0, :]
        self.latitudes = latitudes
        self.longitudes = longitudes

        # save dates times steps and levels
        dates = set()
        times = set()
        steps = set()
        levels = set()
        members = set()
        parameters_full_names = set()
        parameters_short_names = set()
        forecast_dates = set()
        for grib in gribs:
            # filter levels that are not isobaricInhPa
            if not (grib.typeOfLevel == "isobaricInhPa"):
                continue

            dates.add(grib.date)
            times.add(grib.dataTime)
            date = grib.validDate.replace(tzinfo=pytz.utc)
            forecast_dates.add(date.timestamp())

            try:
                int(grib.stepRange)
                steps.add(grib.stepRange)
            except:
                # do nothgin
                continue
            levels.add(grib.level)

            # members.add(grib.perturbationNumber - 1)
            parameters_full_names.add(grib.parameterName)
            parameters_short_names.add(grib.shortName)
        self.dates = list(dates)
        self.times = list(times)
        self.steps = list(steps)
        # check if grib has the content supported by GribPygrib object.
        if len(self.dates) > 1 or len(self.times) > 1:
            sys.exit(
                "Not supported by GribPygrib dates or times len is greater than 1. dates{0} times:{1}".format(
                    self.dates, self.times
                )
            )
        self.forecast_dates = sorted(list(forecast_dates))
        self.levels = sorted(list(levels))
        self.members = sorted(list(members))
        self.parameters_full_names = list(parameters_full_names)
        self.parameters_short_names = list(parameters_short_names)
        self.num_forecast = len(dates) * len(times) * len(steps)
        self.timestamps = sorted(computeTimeStamps(dates, times, steps))

        # evaluate selected variables to be able to specify pl and/or forecast dates
        if selected_forecast_dates is None:
            self.selected_forecast_dates = self.timestamps
        else:
            self.selected_forecast_dates = selected_forecast_dates

        if selected_levels is None:
            self.selected_levels = self.levels
        else:
            self.selected_levels = selected_levels

        # create list to later use index property
        self.selected_forecast_dates = list(self.selected_forecast_dates)
        self.selected_levels = list(self.selected_levels)

    def isParameterInGrib(self, parameter):
        """
        Method to check if a parameter is in the grib file.

        :param parameter: Parameter that needs to be checked.
        :type parameter: str
        :return: True if is in grib, False otherwise.
        :rtype: bool
        """
        result = False
        if parameter in self.parameters_short_names:
            result = True
        return result

    # Generic function to collect parameter values
    # ============================================
    def getParameterUniqueForecast(self, parameter, levels=None):
        gribs = self.gribs
        merged_matrix_dict = {parameter.upper(): {}}
        merged_matrix_dict[parameter.upper()]["values"] = []
        merged_matrix_dict[parameter.upper()]["longs"] = []
        merged_matrix_dict[parameter.upper()]["lats"] = []
        merged_matrix_dict[parameter.upper()]["times"] = []
        if levels is None:
            levels = self.selected_levels
        coordinates = [
            ("t", self.selected_forecast_dates),
            ("pl", levels),
            ("lat", self.latitudes),
            ("lon", self.longitudes),
        ]
        axis_sizes = [len(c_values) for (c_name, c_values) in coordinates]
        nan_array = np.empty(axis_sizes)
        nan_array.fill(np.NaN)
        merged_matrix_dict[parameter.upper()]["values"] = nan_array
        gribs.rewind()
        for grib in gribs:
            ts = computeTimeStamp(grib.date, grib.dataTime, grib.stepRange)
            if (
                (grib.shortName == parameter)
                and (ts in self.selected_forecast_dates)
                and (grib.level in levels)
            ):
                index_time = self.selected_forecast_dates.index(ts)
                if levels is None:
                    index_level = 0
                else:
                    index_level = levels.index(grib.level)
                merged_matrix_dict[parameter.upper()]["values"][
                    index_time, index_level, :, :
                ] = np.array(grib.values)
        merged_matrix_dict[parameter.upper()]["times"] = np.array(
            self.selected_forecast_dates
        )
        merged_matrix_dict[parameter.upper()]["levels"] = np.array(levels)
        merged_matrix_dict[parameter.upper()]["longs"] = np.array(self.longitudes)
        merged_matrix_dict[parameter.upper()]["lats"] = np.array(self.latitudes)
        return merged_matrix_dict

    # parameters needed in convection
    # ===============================
    def getTTs(self):
        """
        This function gets the parameter 'totalx': Total totals index (K).
        It makes use of the method :func:`getParameters`.

        :return: Dictionary with the array containing parameter 'totalx'.
        :rtype: dict
        """
        return self.getParameters(["totalx"])

    def getCPs(self):
        """
        This function gets the parameter 'cp': Convective precipitation.
        It makes use of the method :func:`getParameters`.

        :return: Dictionary with the array containing parameter 'cp'.
        :rtype: dict
        """
        return self.getParameters(["cp"])

    # parameters needed in icing
    # ==========================
    def getTemps(self):
        """
        This function gets the parameter 't': Temperature (K).
        It makes use of the method :func:`getParameters`.

        :return: Dictionary with the array containing parameter 't'.
        :rtype: dict
        """
        return self.getParameters(["t"], self.levels)

    def getRelativeHumidity(self):
        """
        This function gets the parameter 'r': Relative humidity (%).
        It makes use of the method :func:`getParameters`.

        :return: Dictionary with the array containing parameter 'r'.
        :rtype: dict
        """
        return self.getParameters(["r"], self.levels)

    def getOmega(self):
        """
        This function gets the parameter 'w': Vertical velocity (Pa/s).
        It makes use of the method :func:`getParameters`.

        :return: Dictionary with the array containing parameter 'w'.
        :rtype: dict
        """
        return self.getParameters(["w"], self.levels)

    # parameters needed in wind_uncertainty
    # =====================================
    def getUs(self):
        """
        This function gets the parameter 'u': U component of wind (m/s).
        It makes use of the method :func:`getParameters`.

        :return: Dictionary with the array containing parameter 'u'.
        :rtype: dict
        """
        return self.getParameters(["u"], self.levels)

    def getVs(self):
        """
        This function gets the parameter 'v': V component of wind (m/s).
        It makes use of the method :func:`getParameters`.

        :return: Dictionary with the array containing parameter 'v'.
        :rtype: dict
        """
        return self.getParameters(["v"], self.levels)

    # Generic function to collect parameter values
    # ============================================
    def getParameter(self, parameter, level=None):
        gribs = self.gribs
        merged_matrix_dict = {parameter.upper(): {}}
        merged_matrix_dict[parameter.upper()]["values"] = []
        merged_matrix_dict[parameter.upper()]["longs"] = []
        merged_matrix_dict[parameter.upper()]["lats"] = []
        merged_matrix_dict[parameter.upper()]["times"] = []

        if level is None:
            coordinates = [
                ("t", self.selected_forecast_dates),
                ("pl", [-9999]),
                ("ens_n", self.members),
                ("lat", self.latitudes),
                ("lon", self.longitudes),
            ]
            axis_sizes = [len(c_values) for (c_name, c_values) in coordinates]
            nan_array = np.empty(axis_sizes)
            nan_array.fill(np.NaN)
            merged_matrix_dict[parameter.upper()]["values"] = nan_array
            gribs.rewind()
            for grib in gribs:
                ts = computeTimeStamp(grib.date, grib.dataTime, grib.stepRange)
                if (grib.shortName == parameter) and (
                    ts in self.selected_forecast_dates
                ):
                    index_time = self.selected_forecast_dates.index(ts)
                    index_level = 0
                    index_members = self.members.index(grib.perturbationNumber - 1)
                    merged_matrix_dict[parameter.upper()]["values"][
                        index_time, index_level, index_members, :, :
                    ] = np.array(grib.values)
        else:
            coordinates = [
                ("t", self.selected_forecast_dates),
                ("pl", self.selected_levels),
                ("ens_n", self.members),
                ("lat", self.latitudes),
                ("lon", self.longitudes),
            ]
            axis_sizes = [len(c_values) for (c_name, c_values) in coordinates]
            nan_array = np.empty(axis_sizes)
            nan_array.fill(np.NaN)
            merged_matrix_dict[parameter.upper()]["values"] = nan_array
            gribs.rewind()
            for grib in gribs:
                ts = computeTimeStamp(grib.date, grib.dataTime, grib.stepRange)
                if (
                    (grib.shortName == parameter)
                    and (ts in self.selected_forecast_dates)
                    and (grib.level in self.selected_levels)
                ):
                    index_time = self.selected_forecast_dates.index(ts)
                    index_level = self.selected_levels.index(grib.level)
                    index_members = self.members.index(grib.perturbationNumber - 1)
                    merged_matrix_dict[parameter.upper()]["values"][
                        index_time, index_level, index_members, :, :
                    ] = np.array(grib.values)
        merged_matrix_dict[parameter.upper()]["times"] = np.array(
            self.selected_forecast_dates
        )
        merged_matrix_dict[parameter.upper()]["levels"] = np.array(self.selected_levels)
        merged_matrix_dict[parameter.upper()]["longs"] = np.array(self.longitudes)
        merged_matrix_dict[parameter.upper()]["lats"] = np.array(self.latitudes)

        return merged_matrix_dict

    def getParameters(self, parameters, level=None):
        """
        This method returns a dictionary in which each key corresponds to a different parameter from *parameters*. The
        associated value is another dictionary with keys: 'values' (array with the parameter values), 'longs'
        (array of longitudes), 'lats' (array of latitudes), 'levels' (array with the pressure levels) and 'times'
        (array of times).

        :param parameters: Parameters to get from the grib file.
        :type parameters: list
        :param level: List of pressure levels.
        :type level: list
        :return: Dictionary with arrays corresponding to the values of the different parameters.
        :rtype: dict
        """
        gribs = self.gribs
        for var in parameters:
            merged_matrix_dict = {var.upper(): {}}
            merged_matrix_dict[var.upper()]["values"] = []
            merged_matrix_dict[var.upper()]["longs"] = []
            merged_matrix_dict[var.upper()]["lats"] = []
            merged_matrix_dict[var.upper()]["times"] = []

        if level is None:
            coordinates = [
                ("t", self.selected_forecast_dates),
                ("pl", [-9999]),
                ("ens_n", self.members),
                ("lat", self.latitudes),
                ("lon", self.longitudes),
            ]
            axis_sizes = [len(c_values) for (c_name, c_values) in coordinates]
            nan_array = np.empty(axis_sizes)
            nan_array.fill(np.NaN)
            for var in parameters:
                merged_matrix_dict[var.upper()]["values"] = nan_array
            gribs.rewind()
            for grib in gribs:
                ts = computeTimeStamp(grib.date, grib.dataTime, grib.stepRange)
                if (grib.shortName in parameters) and (
                    ts in self.selected_forecast_dates
                ):
                    index_time = self.selected_forecast_dates.index(ts)
                    index_level = 0
                    index_members = self.members.index(grib.perturbationNumber - 1)
                    merged_matrix_dict[grib.shortName.upper()]["values"][
                        index_time, index_level, index_members, :, :
                    ] = np.array(grib.values)
        else:
            coordinates = [
                ("t", self.selected_timestamps),
                ("pl", self.selected_levels),
                ("ens_n", self.members),
                ("lat", self.latitudes),
                ("lon", self.longitudes),
            ]
            axis_sizes = [len(c_values) for (c_name, c_values) in coordinates]
            nan_array = np.empty(axis_sizes)
            nan_array.fill(np.NaN)
            for var in parameters:
                merged_matrix_dict[var.upper()]["values"] = nan_array
            gribs.rewind()
            for grib in gribs:
                ts = computeTimeStamp(grib.date, grib.dataTime, grib.stepRange)
                if (
                    (grib.shortName in parameters)
                    and (ts in self.selected_forecast_dates)
                    and (grib.level in self.selected_levels)
                ):
                    index_time = self.selected_forecast_dates.index(ts)
                    index_level = self.selected_levels.index(grib.level)
                    index_members = self.members.index(grib.perturbationNumber - 1)
                    merged_matrix_dict[grib.shortName.upper()]["values"][
                        index_time, index_level, index_members, :, :
                    ] = np.array(grib.values)

        for var in parameters:
            merged_matrix_dict[var.upper()]["times"] = np.array(
                self.selected_forecast_dates
            )
            merged_matrix_dict[var.upper()]["levels"] = np.array(self.levels)
            merged_matrix_dict[var.upper()]["longs"] = np.array(self.longitudes)
            merged_matrix_dict[var.upper()]["lats"] = np.array(self.latitudes)

        return merged_matrix_dict

    def get_grib_cat_xarray(self):
        """
        This method gets the parameters used in CAT calculations: 'u' (U component of wind), 'v' (V component of wind)
        and 'z' (Geopotential). It makes use of the method :func:`getParameters`.

        :return: Dictionary containing values of parameters 'u', 'v' and 'z'.
        :rtype: dict
        """
        parameters = ["u", "v", "z"]
        levels = self.levels
        return self.getParameters(parameters, levels)

    def get_grib_convection_xarray(self):
        """
        This method gets the parameters used in Convection calculations: 'totalx' (Total totals index), 'cp'
        (convective precipitation). It makes use of the method :func:`getParameters`.

        :return: Dictionary containing values of parameters 'totalx' and 'cp'.
        :rtype: dict
        """
        parameters = ["totalx", "cp"]
        return self.getParameters(parameters)

    def get_grib_icing_xarray(self):
        """
        This method gets the parameters used in Icing calculations: 't' (Temperature), 'r' (Relative humidity)
        and 'w' (Vertical velocity). It makes use of the method :func:`getParameters`.

        :return: Dictionary containing values of parameters 't', 'r' and 'w'.
        :rtype: dict
        """
        parameters = ["t", "r", "w"]
        levels = self.levels
        return self.getParameters(parameters, levels)

    def get_grib_wind_uncertainty_xarray(self):
        """
        This method gets the parameters used in Wind Uncertainty calculations: 'u' (U component of wind), 'v' (
        V component of wind) and 't' (Temperature). It makes use of the method :func:`getParameters`.

        :return: Dictionary containing values of parameters 'u', 'v' and 't'.
        :rtype: dict
        """
        parameters = ["u", "v", "t"]
        levels = self.levels
        return self.getParameters(parameters, levels)

    def get_grib_windshear_xarray(self):
        """
        This method gets the parameters used in Windshear calculations: 'u' (U component of wind), 'v' (V component of wind)
        and 'z' (Geopotential). It makes use of the method :func:`getParameters`.

        :return: Dictionary containing values of parameters 'u', 'v' and 'z'.
        :rtype: dict
        """
        parameters = ["u", "v", "z"]
        levels = self.levels
        return self.getParameters(parameters, levels)

    def get_grib_all_parameters_pl(self):
        """
        This method gets all the parameters used in PL calculations. It makes use of the method :func:`getParameters`.

        :return: Dictionary containing values of all the parameters.
        :rtype: dict
        """
        parameters = self.parameters_short_names
        levels = self.levels
        return self.getParameters(parameters, levels)

    def get_grib_all_parameters_sfc(self):
        """
        This method gets all the parameters used in SFC calculations. It makes use of the method :func:`getParameters`.

        :return: Dictionary containing values of all the parameters.
        :rtype: dict
        """
        parameters = self.parameters_short_names
        return self.getParameters(parameters)

    def get_grib_all_parameters_donuts_pl(self):
        """
        This method gets the parameters used in DONUT'S PL calculations : 'u' (U component of wind), 'v'
        (V component of wind), 'z' (Geopotential), 't' (Temperature), 'r' (Relative humidity)
        and 'w' (Vertical velocity) . It makes use of the method :func:`getParameters`.

        :return: Dictionary containing values of parameters 'u', 'v', 'z', 't', 'r' and 'w'.
        :rtype: dict
        """
        parameters = ["u", "v", "z", "t", "r", "w"]
        levels = self.levels
        return self.getParameters(parameters, levels)

    def get_grib_all_parameters_donuts_sfc(self):
        """
        This method gets the parameters used in DONUT'S SFC calculations : 'totalx' (totals total index) and 'cp'
        (convective precipitation). It makes use of the method :func:`getParameters`.

        :return:  Dictionary containing values of parameters 'totalx' and 'cp'.
        :rtype: dict
        """
        parameters = ["totalx", "cp"]
        return self.getParameters(parameters)
