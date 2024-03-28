import pytest
from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import load_aircraft_engine_params

import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.jet as jet

aircraft_parameters = load_aircraft_engine_params("A320")

T_15 = 288.15
T_0 = 273.15
altitude_current = 34_000
altitude_next = 35_000
delta_time = 30
mach_number = 0.78
pressure_0 = 100_000
wing_surface_area = aircraft_parameters['wing_surface_area']
mtow = aircraft_parameters['amass_mtow']
climb_angle = 0
psi_0 =aircraft_parameters['psi_0']
reynolds_number = 64_412_748
wing_span = aircraft_parameters['wing_span']
wing_aspect_ratio = aircraft_parameters['wing_aspect_ratio']
cos_sweep = aircraft_parameters['cos_sweep']

def test_speed_of_sound():
    assert atm_params.local_speed_of_sound(T_15) == pytest.approx(340.2, rel=1e-3)

def test_dynamic_viscosity():
    assert atm_params.dynamic_viscosity(T_15) == pytest.approx(1.789e-5, rel=1e-2)

def test_air_density():
    assert atm_params.air_density(pressure_0, T_0) == pytest.approx(1.2754, rel=1e-3)

def test_reynolds_number():
    assert atm_params.reynolds_number(wing_span, wing_aspect_ratio, mach_number, T_15, pressure_0) == pytest.approx(64_412_748, rel=1e-3)

def test_rocd():
    assert jet.rate_of_climb_descent(altitude_current, altitude_next, delta_time) == pytest.approx((altitude_next - altitude_current)/(delta_time/60), rel=1e-4)

def test_lift_coefficient():
    assert atm_params.lift_coefficient(wing_surface_area, mtow, pressure_0, T_0, mach_number, climb_angle) == pytest.approx(0.13828, rel=1e-3)

def test_skin_friction_coefficient():
    reynolds_number = atm_params.reynolds_number(wing_span, wing_aspect_ratio, mach_number, T_15, pressure_0)
    assert atm_params.skin_friction_coefficient(reynolds_number) == pytest.approx(0.0269 / (reynolds_number**0.14), rel=1e-5)

def test_zero_lift_drag():
    assert atm_params.zero_lift_drag_coefficient(0.0269 / (reynolds_number**0.14), psi_0) == pytest.approx(0.0269 / (reynolds_number**0.14) * psi_0, rel=1e-5)

def test_oswald_efficiency_factor2():
    reynolds_number = atm_params.reynolds_number(wing_span, wing_aspect_ratio, mach_number, T_15, pressure_0)
    zero_lift_drag = 0.0269 / (reynolds_number**0.14) * psi_0
    assert atm_params.oswald_efficiency_factor(zero_lift_drag, aircraft_parameters) == pytest.approx(0.85, rel=1e-2)

def test_k1():
    zero_lift_drag = 0.0269 / (reynolds_number**0.14) * psi_0
    assert atm_params._non_vortex_lift_dependent_drag_factor(zero_lift_drag, cos_sweep) == pytest.approx(0.80 * (1 - 0.53 * cos_sweep) * zero_lift_drag, rel=1e-2)

def test_wave_drag_coefficient():
    c_lift = atm_params.lift_coefficient(wing_surface_area, mtow, pressure_0, T_0, mach_number, climb_angle)
    assert atm_params.wave_drag_coefficient(mach_number, c_lift, aircraft_parameters) == pytest.approx(0.00049245, rel=1e-2)

def test_airframe_drag_coefficient():
    c_lift = atm_params.lift_coefficient(wing_surface_area, mtow, pressure_0, T_0, mach_number, climb_angle)
    zero_lift_drag = atm_params.zero_lift_drag_coefficient(0.0269 / (reynolds_number**0.14), psi_0)
    wave_drag = atm_params.wave_drag_coefficient(mach_number, c_lift, aircraft_parameters)
    oswald_efficiency = atm_params.oswald_efficiency_factor(zero_lift_drag, aircraft_parameters)
    assert atm_params.airframe_drag_coefficient(zero_lift_drag, wave_drag, c_lift, oswald_efficiency, wing_aspect_ratio) == pytest.approx(0.019465667160460293, rel=1e-2)

# @pytest.mark.parametrize("wing_surface_area, mach_num, air_temperature, air_pressure, expected", [
#     (10, 0.8, 300, 101325, 1.0),
#     (10, 0.8, 300, 101325, 1.0),
#     (10, 0.8, 300, 101325, 1.0),
# ])
# def test_reynolds_number(wing_surface_area, mach_num, air_temperature, air_pressure, expected):
#     assert atm_params.reynolds_number(wing_surface_area, mach_num, air_temperature, air_pressure) == pytest.approx(expected, rel=1e-4)