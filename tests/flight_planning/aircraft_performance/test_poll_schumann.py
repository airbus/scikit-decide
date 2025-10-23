import sys

import pytest


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="pygrib does not install on windows"
)
def test_drag():
    import numpy as np
    from openap.extra.aero import ft, kts, mach2tas

    from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
        AircraftState,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.bean.atmos_isa import (
        temperature,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.service.aerodynamics_service import (
        AerodynamicsService,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
        PerformanceModelEnum,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.performance.phase_enum import (
        PhaseEnum,
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

    atmosphere_service = AtmosphereService()
    poll_schumann_aero = AerodynamicsService()
    poll_schumann_settings = poll_schumann_aero.init_settings(
        "A320", PerformanceModelEnum.POLL_SCHUMANN
    )

    test_cases = [
        {
            "zp_ft": 34000,
            "expected_cl": 0.475,
            "expected_cd": 0.0285,
            "dISA": 220.79 - temperature(34000, disa=0),
        },
        {
            "zp_ft": 41450,
            "expected_cl": 0.679,
            "expected_cd": 0.0402,
            "dISA": 216.65 - temperature(34000, disa=0),
        },
    ]

    for case in test_cases:
        acState = AircraftState(
            model_type="A320",
            performance_model_type=PerformanceModelEnum.OPENAP,
            gw_kg=58800.0,
            zp_ft=case["zp_ft"],
            mach=0.753,
            phase=PhaseEnum.CRUISE,
            rating_level=RatingEnum.MCL,
            cg=0.3,
            gamma_air_deg=0,
        )

        isa_atmosphere_settings = IsaAtmosphereSettings(d_isa=case["dISA"])

        weather_state = atmosphere_service.retrieve_weather_state(
            atmosphere_settings=isa_atmosphere_settings, four_dimensions_state=acState
        )

        acState.weather_state = weather_state

        delta = acState.weather_state.static_pressure_pa / 101325.0

        acState.cl = (2 * acState.gw_kg * 9.80665) / (
            delta * 101325.0 * 1.4 * poll_schumann_settings.sref * acState.mach**2
        )

        np.testing.assert_almost_equal(acState.cl, case["expected_cl"], decimal=2)

        val_skdecide = poll_schumann_aero.compute_drag_coefficient(
            aerodynamics_settings=poll_schumann_settings, aircraft_state=acState
        )

        np.testing.assert_almost_equal(val_skdecide, case["expected_cd"], decimal=4)


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="pygrib does not install on windows"
)
def test_thrust():
    import numpy as np
    from openap.extra.aero import ft, kts, mach2tas

    from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
        AircraftState,
    )
    from skdecide.hub.domain.flight_planning.aircraft_performance.bean.atmos_isa import (
        temperature,
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

    atmosphere_service = AtmosphereService()
    poll_schumann_propu = PropulsionService()
    poll_schumann_settings = poll_schumann_propu.init_settings(
        "A320", PerformanceModelEnum.POLL_SCHUMANN
    )

    test_cases = [
        {
            "zp_ft": 34000,
            "expected_cl": 0.475,
            "expected_fn": 3.4638,
            "expected_ff": 0.593,
            "dISA": 220.79 - temperature(34000, disa=0),
        },
        {
            "zp_ft": 41450,
            "expected_cl": 0.679,
            "expected_fn": 3.4156,
            "expected_ff": 0.5710,
            "dISA": 216.65 - temperature(34000, disa=0),
        },
    ]

    for case in test_cases:
        acState = AircraftState(
            model_type="A320",
            performance_model_type=PerformanceModelEnum.OPENAP,
            gw_kg=58800.0,
            zp_ft=case["zp_ft"],
            mach=0.753,
            phase=PhaseEnum.CRUISE,
            rating_level=RatingEnum.MCL,
            cg=0.3,
            gamma_air_deg=0,
        )

        isa_atmosphere_settings = IsaAtmosphereSettings(d_isa=case["dISA"])

        weather_state = atmosphere_service.retrieve_weather_state(
            atmosphere_settings=isa_atmosphere_settings, four_dimensions_state=acState
        )

        acState.weather_state = weather_state

        delta = acState.weather_state.static_pressure_pa / 101325.0

        acState.cl = (2 * acState.gw_kg * 9.80665) / (
            delta * 101325.0 * 1.4 * poll_schumann_settings.sref * acState.mach**2
        )

        np.testing.assert_almost_equal(acState.cl, case["expected_cl"], decimal=2)

        val_skdecide = poll_schumann_propu.compute_total_net_thrust_n(
            propulsion_settings=poll_schumann_settings, aircraft_state=acState
        )

        np.testing.assert_almost_equal(
            val_skdecide / 1e4, case["expected_fn"], decimal=2
        )

        val_skdecide = poll_schumann_propu.compute_total_fuel_flow_kg_per_sec(
            propulsion_settings=poll_schumann_settings, aircraft_state=acState
        )

        np.testing.assert_almost_equal(val_skdecide, case["expected_ff"], decimal=3)
