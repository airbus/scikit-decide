import calendar
import collections
import datetime as datetime
import logging
import os
import urllib.request as request
from functools import reduce
from typing import List, Optional

import numpy as np

from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools import (
    std_atm as standard_atmosphere,
)
from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.interpolator.GenericInterpolator import (
    GenericWindInterpolator,
)
from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.parser_pygrib import (
    GribPygribUniqueForecast,
)
from skdecide.utils import get_data_home

logger = logging.getLogger(__name__)


def tree():
    return collections.defaultdict(tree)


def get_absolute_path(filename, relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(filename), relative_path))


def create_merged_matrix(list_files: List[str], params: Optional[List[str]]):
    if params is None:
        params = ["u", "v", "t", "r"]

    def get_params(f, params_to_retrieve):
        a = GribPygribUniqueForecast(
            grib_path=os.path.dirname(f), grib_name=os.path.basename(f)
        )
        mats = list(
            map(
                lambda x: a.getParameterUniqueForecast(parameter=x, levels=None),
                params_to_retrieve,
            )
        )
        m = {}
        list(map(lambda x: m.update(x), mats))
        del a
        return m

    def merge(list_matrix):
        merged_matrix = {k: {} for k in list_matrix[0]}
        for k in merged_matrix:
            for key in ["longs", "lats", "levels"]:
                merged_matrix[k][key] = list_matrix[0][k][key]
            merged_matrix[k]["times"] = []
            merged_matrix[k]["values"] = []
        for m in list_matrix:
            for k in merged_matrix:
                merged_matrix[k]["times"] += list(m[k]["times"])
                merged_matrix[k]["values"] += [m[k]["values"][0]]
        for k in merged_matrix:
            merged_matrix[k]["times"] = np.array(merged_matrix[k]["times"])
            merged_matrix[k]["values"] = np.array(merged_matrix[k]["values"])
        return merged_matrix

    result = merge(
        list(map(lambda x: get_params(x, params_to_retrieve=params), list_files))
    )
    return result


def get_weather_matrix(
    year,
    month,
    day,
    forecast,
    save_to_npz=True,
    delete_grib_from_local=True,
    delete_npz_from_local=False,
    download_grib=True,
):
    """
    :param year:
    :param month:
    :param day:
    :param forecast:
    :return:
    """
    exportdir_grib = get_absolute_path(
        __file__,
        f"{get_data_home()}/weather/grib/"
        + forecast
        + "/"
        + str(year)
        + str(month)
        + str(day),
    )
    exportdir_npz = get_absolute_path(
        __file__,
        f"{get_data_home()}/weather/npz/"
        + forecast
        + "/"
        + str(year)
        + str(month)
        + str(day),
    )
    if not os.path.exists(exportdir_npz):
        os.makedirs(exportdir_npz)
    list_files = [
        os.path.join(exportdir_npz, x) for x in os.listdir(exportdir_npz) if "npz" in x
    ]
    if len(list_files) > 0:
        # In case you already have an npz locally
        logger.info("You have the npz on your local computer")
        p = np.load(list_files[0], allow_pickle=True)
        if delete_npz_from_local:
            os.remove(list_files[0])
        return p
    # The npz is not on S3 neither locally...
    if not os.path.exists(exportdir_grib):
        os.makedirs(exportdir_grib)
    list_files = [os.path.join(exportdir_grib, x) for x in os.listdir(exportdir_grib)]
    if len(list_files) >= 4:
        logger.info("Grib found locally")
    if len(list_files) == 0:
        if not download_grib:
            return {}
        files, address = UrlGeneratorWithForecastLayer.get_list_of_url_forecast(
            day=day, month=month, year=year, forecast=forecast
        )
        list_files = []
        for i in range(len(files)):
            print("Downloading : ", address[i])
            try:
                request.urlretrieve(
                    address[i], filename=os.path.join(exportdir_grib, files[i])
                )
                list_files += [os.path.join(exportdir_grib, files[i])]
            except Exception as e:
                print(e)
                pass
        list_files = [
            os.path.join(exportdir_grib, x) for x in os.listdir(exportdir_grib)
        ]
        if not len(list_files) > 0:
            return {}
    matrix = create_merged_matrix(list_files=list_files, params=["u", "v", "t", "r"])
    if save_to_npz:
        file_output = os.path.join(
            exportdir_npz, str(year) + str(month) + str(day) + ".npz"
        )
        if not os.path.exists(os.path.dirname(file_output)):
            if not os.path.dirname(file_output) == "":
                os.makedirs(os.path.dirname(file_output))
        np.savez_compressed(file_output, **matrix)
    if delete_grib_from_local:
        for l in list_files:
            os.remove(l)
    if delete_npz_from_local:
        os.remove(
            os.path.join(exportdir_npz, str(year) + str(month) + str(day) + ".npz")
        )

    return matrix


class UrlGeneratorWithForecastLayer:
    @staticmethod
    def forecast_time(x):
        return x[-13:-9]

    @staticmethod
    def get_date(x):
        d = datetime.date(x[-22:-18], x[-18:-16], x[-16:-14])
        return d

    @staticmethod
    def get_folder_name(x):
        return x[-22:-14]

    @staticmethod
    def get_list_of_url(day, month, year, hours_prediction, hours_from_prediction):
        files = [
            "gfs_4_"
            + year
            + month
            + day
            + "_"
            + hour_prediction
            + "_"
            + hour_from_prediction
            + ".grb2"
            for hour_prediction, hour_from_prediction in zip(
                hours_prediction, hours_from_prediction
            )
        ]
        return files, [
            "https://www.ncei.noaa.gov/data/global-forecast-system/access/grid-004-0.5-degree/forecast/"
            + year
            + month
            + "/"
            + year
            + month
            + day
            + "/"
            + file
            for file in files
        ]

    @staticmethod
    def return_list_of_url():
        years = ["2016", "2017", "2018"]
        months = [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ]
        all = [("31", "12", "2015")] + [
            ("0" + str(d) if d < 10 else str(d), m, y)
            for y in years
            for m in months
            for d in range(1, calendar.monthrange(int(y), int(m))[1] + 1)
        ]
        hours_prediction_0 = ["0000", "0000", "0000", "0000"]
        hours_prediction_6 = ["0600", "0600", "0600", "0600"]
        hours_prediction_12 = ["1200", "1200", "1200", "1200"]
        hours_prediction_18 = ["1800", "1800", "1800", "1800"]
        hours_from_prediction = ["000", "006", "012", "018"]
        hours_prediction_obs = ["0000", "0600", "1200", "1800"]
        hours_from_prediction_obs = ["000", "000", "000", "000"]
        dict_files = {}
        dict_files["forecast__0000"] = reduce(
            lambda x, y: [
                x[0]
                + UrlGeneratorWithForecastLayer.get_list_of_url(
                    y[0], y[1], y[2], hours_prediction_0, hours_from_prediction
                )[0],
                x[1]
                + UrlGeneratorWithForecastLayer.get_list_of_url(
                    y[0], y[1], y[2], hours_prediction_0, hours_from_prediction
                )[1],
            ],
            all,
            [[], []],
        )[0][4:]
        dict_files["forecast__0600"] = reduce(
            lambda x, y: [
                x[0]
                + UrlGeneratorWithForecastLayer.get_list_of_url(
                    y[0], y[1], y[2], hours_prediction_6, hours_from_prediction
                )[0],
                x[1]
                + UrlGeneratorWithForecastLayer.get_list_of_url(
                    y[0], y[1], y[2], hours_prediction_6, hours_from_prediction
                )[1],
            ],
            all,
            [[], []],
        )[0][3:-1]
        dict_files["forecast__1200"] = reduce(
            lambda x, y: [
                x[0]
                + UrlGeneratorWithForecastLayer.get_list_of_url(
                    y[0], y[1], y[2], hours_prediction_12, hours_from_prediction
                )[0],
                x[1]
                + UrlGeneratorWithForecastLayer.get_list_of_url(
                    y[0], y[1], y[2], hours_prediction_12, hours_from_prediction
                )[1],
            ],
            all,
            [[], []],
        )[0][2:-2]
        dict_files["forecast__1800"] = reduce(
            lambda x, y: [
                x[0]
                + UrlGeneratorWithForecastLayer.get_list_of_url(
                    y[0], y[1], y[2], hours_prediction_18, hours_from_prediction
                )[0],
                x[1]
                + UrlGeneratorWithForecastLayer.get_list_of_url(
                    y[0], y[1], y[2], hours_prediction_18, hours_from_prediction
                )[1],
            ],
            all,
            [[], []],
        )[0][1:-3]

        dict_files["nowcast"] = reduce(
            lambda x, y: [
                x[0]
                + UrlGeneratorWithForecastLayer.get_list_of_url(
                    y[0], y[1], y[2], hours_prediction_obs, hours_from_prediction_obs
                )[0],
                x[1]
                + UrlGeneratorWithForecastLayer.get_list_of_url(
                    y[0], y[1], y[2], hours_prediction_obs, hours_from_prediction_obs
                )[1],
            ],
            all,
            [[], []],
        )[0][4:]
        for k in dict_files:
            print(dict_files[k])
            print(len(dict_files[k]))

    @staticmethod
    def get_list_of_url_forecast(day, month, year, forecast):
        years = [year]
        months = [month]
        all = [
            ("0" + str(d) if d < 10 else str(d), m, y)
            for y in years
            for m in months
            for d in [int(day)]
        ]
        if int(day) == 1:
            if int(month) == 1:
                add = [("31", "12", str(int(year) - 1))]
            else:
                add = [
                    (
                        str(calendar.monthrange(int(year), int(month) - 1)[1]),
                        str(int(month) - 1),
                        str(year),
                    )
                ]
        else:
            add = [
                (
                    "0" + str(int(day) - 1) if int(day) - 1 < 10 else str(int(day) - 1),
                    str(month),
                    str(year),
                )
            ]
        all = add + all
        hours_prediction_0 = ["0000", "0000", "0000", "0000"]
        hours_prediction_6 = ["0600", "0600", "0600", "0600"]
        hours_prediction_12 = ["1200", "1200", "1200", "1200"]
        hours_prediction_18 = ["1800", "1800", "1800", "1800"]
        hours_from_prediction = ["000", "006", "012", "018"]
        hours_prediction_obs = ["0000", "0600", "1200", "1800"]
        hours_from_prediction_obs = ["000", "000", "000", "000"]
        if forecast == "forecast__0000":
            p = reduce(
                lambda x, y: [
                    x[0]
                    + UrlGeneratorWithForecastLayer.get_list_of_url(
                        y[0], y[1], y[2], hours_prediction_0, hours_from_prediction
                    )[0],
                    x[1]
                    + UrlGeneratorWithForecastLayer.get_list_of_url(
                        y[0], y[1], y[2], hours_prediction_0, hours_from_prediction
                    )[1],
                ],
                all,
                [[], []],
            )
            return p[0][4:], p[1][4:]
        if forecast == "forecast__0600":
            p = reduce(
                lambda x, y: [
                    x[0]
                    + UrlGeneratorWithForecastLayer.get_list_of_url(
                        y[0], y[1], y[2], hours_prediction_6, hours_from_prediction
                    )[0],
                    x[1]
                    + UrlGeneratorWithForecastLayer.get_list_of_url(
                        y[0], y[1], y[2], hours_prediction_6, hours_from_prediction
                    )[1],
                ],
                all,
                [[], []],
            )
            return p[0][3:-1], p[1][3:-1]
        if forecast == "forecast__1200":
            p = reduce(
                lambda x, y: [
                    x[0]
                    + UrlGeneratorWithForecastLayer.get_list_of_url(
                        y[0], y[1], y[2], hours_prediction_12, hours_from_prediction
                    )[0],
                    x[1]
                    + UrlGeneratorWithForecastLayer.get_list_of_url(
                        y[0], y[1], y[2], hours_prediction_12, hours_from_prediction
                    )[1],
                ],
                all,
                [[], []],
            )
            return p[0][2:-2], p[1][2:-2]
        if forecast == "forecast__1800":
            p = reduce(
                lambda x, y: [
                    x[0]
                    + UrlGeneratorWithForecastLayer.get_list_of_url(
                        y[0], y[1], y[2], hours_prediction_18, hours_from_prediction
                    )[0],
                    x[1]
                    + UrlGeneratorWithForecastLayer.get_list_of_url(
                        y[0], y[1], y[2], hours_prediction_18, hours_from_prediction
                    )[1],
                ],
                all,
                [[], []],
            )
            return p[0][1:-3], p[1][1:-3]
        if forecast == "nowcast":
            p = reduce(
                lambda x, y: [
                    x[0]
                    + UrlGeneratorWithForecastLayer.get_list_of_url(
                        y[0],
                        y[1],
                        y[2],
                        hours_prediction_obs,
                        hours_from_prediction_obs,
                    )[0],
                    x[1]
                    + UrlGeneratorWithForecastLayer.get_list_of_url(
                        y[0],
                        y[1],
                        y[2],
                        hours_prediction_obs,
                        hours_from_prediction_obs,
                    )[1],
                ],
                all,
                [[], []],
            )
            return p[0][4:], p[1][4:]

    @staticmethod
    def get_ensemble_noaa_forecast(day, month, year, forecast, id_forecast):
        hours_from_prediction = ["000", "006", "012", "018", "024"]
        if int(id_forecast) < 10:
            id_forecast_t = "0" + id_forecast
        else:
            id_forecast_t = str(id_forecast)
        files = [
            "gens-a_2_"
            + year
            + month
            + day
            + "_"
            + forecast
            + "_"
            + forc
            + "_"
            + id_forecast_t
            + ".grb2"
            for forc in hours_from_prediction
        ]
        files = [
            "gens-b_3_"
            + year
            + month
            + day
            + "_"
            + forecast
            + "_"
            + forc
            + "_"
            + id_forecast_t
            + ".grb2"
            for forc in hours_from_prediction
        ]
        # return files, ['https://nomads.ncdc.noaa.gov/data/gens/' + year + month + '/' + year + month + day + '/' + file
        #               for file in files]
        # https: // www.ncei.noaa.gov / thredds / dodsC / model - gefs - 003 / 202009 / 20200923 / gensanl - b_3_20200923_0600_000_20.grb2
        return files, [
            "https://www.ncei.noaa.gov/data/global-ensemble-forecast-system/access/1.0-degree-grid/"
            + year
            + month
            + "/"
            + year
            + month
            + day
            + "/"
            + file
            for file in files
        ]
