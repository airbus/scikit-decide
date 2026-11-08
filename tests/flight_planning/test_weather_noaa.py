import os
import urllib.error

import pytest

from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools import (
    get_weather_noaa,
)
from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.get_weather_noaa import (
    GFS_AWS_BASE_URL,
    UrlGeneratorWithForecastLayer,
    get_weather_matrix,
)


def test_gfs_urls_point_to_aws_nodd_bucket():
    files, urls = UrlGeneratorWithForecastLayer.get_list_of_url_forecast(
        day="02", month="07", year="2026", forecast="forecast__0600"
    )
    assert len(files) == len(urls) == 4
    # 0600 run: last step of previous day's run, then the 3 first steps of the day.
    assert urls == [
        f"{GFS_AWS_BASE_URL}/gfs.20260701/06/atmos/gfs.t06z.pgrb2.0p50.f018",
        f"{GFS_AWS_BASE_URL}/gfs.20260702/06/atmos/gfs.t06z.pgrb2.0p50.f000",
        f"{GFS_AWS_BASE_URL}/gfs.20260702/06/atmos/gfs.t06z.pgrb2.0p50.f006",
        f"{GFS_AWS_BASE_URL}/gfs.20260702/06/atmos/gfs.t06z.pgrb2.0p50.f012",
    ]
    assert not any("ncei.noaa.gov" in u for u in urls)
    # local names must be unique
    assert len(set(files)) == len(files)


@pytest.mark.parametrize(
    "forecast", ["forecast__0000", "forecast__0600", "forecast__1200", "forecast__1800"]
)
def test_gfs_urls_first_day_of_month(forecast):
    # previous day belongs to the previous month, which must be zero padded
    _, urls = UrlGeneratorWithForecastLayer.get_list_of_url_forecast(
        day="01", month="07", year="2026", forecast=forecast
    )
    assert len(urls) == 4
    assert all("/gfs.2026" in u for u in urls)
    assert all("/gfs.2026630/" not in u for u in urls)  # unpadded month
    if forecast != "forecast__0000":
        assert any("/gfs.20260630/" in u for u in urls)


def test_get_weather_matrix_raises_when_download_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("SKDECIDE_DATA", str(tmp_path))

    def failing_urlretrieve(url, filename=None):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(get_weather_noaa.request, "urlretrieve", failing_urlretrieve)
    with pytest.raises(RuntimeError, match="Could not download any GFS weather file"):
        get_weather_matrix(year="2026", month="07", day="02", forecast="forecast__0600")


def test_get_weather_matrix_removes_partial_files(tmp_path, monkeypatch):
    monkeypatch.setenv("SKDECIDE_DATA", str(tmp_path))

    def partial_urlretrieve(url, filename=None):
        # write a truncated file, then fail
        with open(filename, "wb") as f:
            f.write(b"x")
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(get_weather_noaa.request, "urlretrieve", partial_urlretrieve)
    with pytest.raises(RuntimeError):
        get_weather_matrix(year="2026", month="07", day="02", forecast="forecast__0600")
    grib_dirs = [d for d, _, fs in os.walk(tmp_path / "weather" / "grib") if fs]
    assert grib_dirs == []


def test_get_weather_matrix_no_download_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SKDECIDE_DATA", str(tmp_path))
    assert (
        get_weather_matrix(
            year="2026",
            month="07",
            day="02",
            forecast="forecast__0600",
            download_grib=False,
        )
        == {}
    )
