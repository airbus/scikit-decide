import sys

import pytest


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="cartopy requires python3.9 or higher"
)
@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="pygrib does not install on windows"
)
def test_thrust():
    from itertools import product

    import numpy as np
    from openap.extra.aero import ft, kts, mach2tas
    from openap.thrust import Thrust

    from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
        AircraftState,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
        PerformanceModelEnum,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.performance.phase_enum import (
        PhaseEnum,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.performance.propulsion.service.propulsion_service import (
        PropulsionService,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.performance.rating_enum import (
        RatingEnum,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.weather.service.atmosphere_service import (
        AtmosphereService,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.weather.settings.isa_atmosphere_settings import (
        IsaAtmosphereSettings,
    )

    openap_propu = PropulsionService()
    openap_settings = openap_propu.init_settings("A320", PerformanceModelEnum.OPENAP)
    openap_thrust = Thrust(ac="A320")

    atmosphere_service = AtmosphereService()

    list_zp_ft = [10000, 20000, 30000, 40000]
    list_gw_kg = [50_000, 60_000, 70_000, 80_000, 90_000, 100_000]
    list_mach = [0.6, 0.7, 0.8, 0.9]
    list_disa = [-10, 0, 10]

    list_test = list(product(list_zp_ft, list_gw_kg, list_mach, list_disa))

    for zp_ft, gw_kg, mach, disa in list_test:
        # initial state of the aircraft
        acState = AircraftState(
            model_type="A320",  # only for OPENAP and POLL_SCHUMANN
            performance_model_type=PerformanceModelEnum.OPENAP,  # PerformanceModelEnum.OPENAP, PerformanceModelEnum.BADA
            gw_kg=gw_kg,
            zp_ft=zp_ft,
            mach_cruise=mach,
            cas_climb_kts=170,
            cas_descent_kts=250,
            phase=PhaseEnum.CRUISE,
            rating_level=RatingEnum.MCL,
            cg=0.3,
            gamma_air_deg=0,
        )

        # current state
        acState.weather_state = atmosphere_service.retrieve_weather_state(
            atmosphere_settings=IsaAtmosphereSettings(d_isa=disa),
            four_dimensions_state=acState,
        )
        acState.mach = mach

        # compute
        val_skdecide = openap_propu.compute_total_net_thrust_n(
            propulsion_settings=openap_settings, aircraft_state=acState
        )

        val_true = openap_thrust.cruise(
            tas=mach2tas(acState.mach, acState.zp_ft * ft) / kts,
            alt=acState.zp_ft,
        )

        np.testing.assert_almost_equal(val_skdecide, val_true, decimal=1)


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="cartopy requires python3.9 or higher"
)
@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="pygrib does not install on windows"
)
def test_fuelflow():
    from itertools import product

    import numpy as np
    from openap.extra.aero import ft, kts, mach2tas
    from openap.fuel import FuelFlow

    from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
        AircraftState,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
        PerformanceModelEnum,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.performance.phase_enum import (
        PhaseEnum,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.performance.propulsion.service.propulsion_service import (
        PropulsionService,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.performance.rating_enum import (
        RatingEnum,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.weather.service.atmosphere_service import (
        AtmosphereService,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.weather.settings.isa_atmosphere_settings import (
        IsaAtmosphereSettings,
    )

    openap_propu = PropulsionService()
    openap_settings = openap_propu.init_settings("A320", PerformanceModelEnum.OPENAP)
    openap_fuelflow = FuelFlow(ac="A320")

    atmosphere_service = AtmosphereService()

    list_zp_ft = [10000, 20000, 30000, 40000]
    list_gw_kg = [50_000, 60_000, 70_000, 80_000, 90_000, 100_000]
    list_mach = [0.6, 0.7, 0.8, 0.9]
    list_disa = [-10, 0, 10]

    list_test = list(product(list_zp_ft, list_gw_kg, list_mach, list_disa))

    for zp_ft, gw_kg, mach, disa in list_test:
        # initial state of the aircraft
        acState = AircraftState(
            model_type="A320",  # only for OPENAP and POLL_SCHUMANN
            performance_model_type=PerformanceModelEnum.OPENAP,  # PerformanceModelEnum.OPENAP, PerformanceModelEnum.BADA
            gw_kg=gw_kg,
            zp_ft=zp_ft,
            mach_cruise=mach,
            cas_climb_kts=170,
            cas_descent_kts=250,
            phase=PhaseEnum.CRUISE,
            rating_level=RatingEnum.MCL,
            cg=0.3,
            gamma_air_deg=0,
        )

        # current state
        acState.weather_state = atmosphere_service.retrieve_weather_state(
            atmosphere_settings=IsaAtmosphereSettings(d_isa=disa),
            four_dimensions_state=acState,
        )
        acState.mach = mach

        # compute
        val_skdecide = openap_propu.compute_total_fuel_flow_kg_per_sec(
            propulsion_settings=openap_settings, aircraft_state=acState
        )

        val_true = openap_fuelflow.enroute(
            mass=acState.gw_kg,
            tas=mach2tas(acState.mach, acState.zp_ft * ft) / kts,
            alt=acState.zp_ft,
            vs=0,
        )

        np.testing.assert_almost_equal(val_skdecide, val_true, decimal=1)
