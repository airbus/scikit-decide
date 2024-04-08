import pytest
import numpy as np
import sys

@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="cartopy requires python3.9 or higher"
)
@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="pygrib does not install on windows"
)

def test_speed_of_sound():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    T_15 = 288.15
    assert atm_params.local_speed_of_sound(T_15) == pytest.approx(340.2, rel=1e-3)

def test_dynamic_viscosity():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    T_15 = 288.15
    assert atm_params.dynamic_viscosity(T_15) == pytest.approx(1.789e-5, rel=1e-2)

def test_air_density():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    T_0 = 273.15
    pressure_0 = 100_000
    assert atm_params.air_density(pressure_0, T_0) == pytest.approx(1.2754, rel=1e-3)

def test_reynolds_number():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import load_aircraft_engine_params
    T_15 = 288.15
    aircraft_parameters = load_aircraft_engine_params("A320")
    wing_aspect_ratio = aircraft_parameters['wing_aspect_ratio']
    wing_span = aircraft_parameters['wing_span']
    mach_number = 0.78
    pressure_0 = 100_000
    assert atm_params.reynolds_number(wing_span, wing_aspect_ratio, mach_number, T_15, pressure_0) == pytest.approx(64_412_748, rel=1e-3)

def test_rocd():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.jet as jet
    altitude_current = 34_000
    altitude_next = 35_000
    delta_time = 30
    assert jet.rate_of_climb_descent(altitude_current, altitude_next, delta_time) == pytest.approx((altitude_next - altitude_current)/(delta_time/60), rel=1e-4)

def test_lift_coefficient():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import load_aircraft_engine_params
    aircraft_parameters = load_aircraft_engine_params("A320")
    wing_surface_area = aircraft_parameters['wing_surface_area']
    mtow = aircraft_parameters['amass_mtow']
    T_0 = 273.15
    mach_number = 0.78
    pressure_0 = 100_000
    climb_angle = 0
    assert atm_params.lift_coefficient(wing_surface_area, mtow, pressure_0, T_0, mach_number, climb_angle) == pytest.approx(0.13828, rel=1e-3)

def test_skin_friction_coefficient():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import load_aircraft_engine_params
    aircraft_parameters = load_aircraft_engine_params("A320")
    wing_aspect_ratio = aircraft_parameters['wing_aspect_ratio']
    wing_span = aircraft_parameters['wing_span']
    T_15 = 288.15
    mach_number = 0.78
    pressure_0 = 100_000
    reynolds_number = atm_params.reynolds_number(wing_span, wing_aspect_ratio, mach_number, T_15, pressure_0)
    assert atm_params.skin_friction_coefficient(reynolds_number) == pytest.approx(0.0269 / (reynolds_number**0.14), rel=1e-5)

def test_zero_lift_drag():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import load_aircraft_engine_params
    aircraft_parameters = load_aircraft_engine_params("A320")
    psi_0 =aircraft_parameters['psi_0']
    reynolds_number = 64_412_748
    assert atm_params.zero_lift_drag_coefficient(0.0269 / (reynolds_number**0.14), psi_0) == pytest.approx(0.0269 / (reynolds_number**0.14) * psi_0, rel=1e-5)

def test_oswald_efficiency_factor2():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import load_aircraft_engine_params
    aircraft_parameters = load_aircraft_engine_params("A320")
    wing_aspect_ratio = aircraft_parameters['wing_aspect_ratio']
    wing_span = aircraft_parameters['wing_span']
    psi_0 =aircraft_parameters['psi_0']
    T_15 = 288.15
    mach_number = 0.78
    pressure_0 = 100_000
    reynolds_number = atm_params.reynolds_number(wing_span, wing_aspect_ratio, mach_number, T_15, pressure_0)
    zero_lift_drag = 0.0269 / (reynolds_number**0.14) * psi_0
    assert atm_params.oswald_efficiency_factor(zero_lift_drag, aircraft_parameters) == pytest.approx(0.85, rel=1e-2)

def test_k1():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import load_aircraft_engine_params
    aircraft_parameters = load_aircraft_engine_params("A320")
    psi_0 =aircraft_parameters['psi_0']
    reynolds_number = 64_412_748
    zero_lift_drag = 0.0269 / (reynolds_number**0.14) * psi_0
    cos_sweep = aircraft_parameters['cos_sweep']
    assert atm_params._non_vortex_lift_dependent_drag_factor(zero_lift_drag, cos_sweep) == pytest.approx(0.80 * (1 - 0.53 * cos_sweep) * zero_lift_drag, rel=1e-2)

def test_wave_drag_coefficient():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import load_aircraft_engine_params
    aircraft_parameters = load_aircraft_engine_params("A320")
    c_lift =0.13828
    mach_number = 0.78
    assert atm_params.wave_drag_coefficient(mach_number, c_lift, aircraft_parameters) == pytest.approx(0.00049245, rel=1e-2)

def test_airframe_drag_coefficient():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import load_aircraft_engine_params
    aircraft_parameters = load_aircraft_engine_params("A320")
    c_lift = 0.13828
    psi_0 = aircraft_parameters['psi_0']
    reynolds_number = 64_412_748
    zero_lift_drag = atm_params.zero_lift_drag_coefficient(0.0269 / (reynolds_number**0.14), psi_0)
    wave_drag = 0.00049245
    oswald_efficiency = atm_params.oswald_efficiency_factor(zero_lift_drag, aircraft_parameters)
    wing_aspect_ratio = aircraft_parameters['wing_aspect_ratio']
    assert atm_params.airframe_drag_coefficient(zero_lift_drag, wave_drag, c_lift, oswald_efficiency, wing_aspect_ratio) == pytest.approx(0.019465667160460293, rel=1e-2)

def test_thrust_force():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import load_aircraft_engine_params
    aircraft_parameters = load_aircraft_engine_params("A320")
    c_lift = 0.13828
    c_drag = 0.019465667160460293
    mtow = aircraft_parameters['amass_mtow']
    climb_angle = 0
    assert atm_params.thrust_force(mtow, c_lift, c_drag, 0, climb_angle) == pytest.approx(101470, rel=1e-2)

def test_engine_thrust_coefficient():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import load_aircraft_engine_params
    f_thrust = 101470
    aircraft_parameters = load_aircraft_engine_params("A320")
    wing_surface_area = aircraft_parameters['wing_surface_area']
    mach_number = 0.78
    pressure_0 = 100_000
    assert atm_params.engine_thrust_coefficient(f_thrust, mach_number, pressure_0, wing_surface_area) == pytest.approx(0.01946566716046029, rel=1e-1)

def test_thrust_coefficient_at_max_efficiency():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import load_aircraft_engine_params
    aircraft_parameters = load_aircraft_engine_params("A320")
    mach_number = 0.78
    assert atm_params.thrust_coefficient_at_max_efficiency(mach_number, aircraft_parameters['m_des'], aircraft_parameters['c_t_des']) == pytest.approx(0.03267886563990706, rel=1e-1)

def test_max_available_thrust_coeff():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.operational_limits as op_lim
    from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import load_aircraft_engine_params
    aircraft_parameters = load_aircraft_engine_params("A320")
    c_t_eta_b = 0.03267886563990706
    T_15 = 288.15
    mach_number = 0.78
    assert op_lim.max_available_thrust_coefficient(T_15, mach_number, c_t_eta_b, aircraft_parameters) == pytest.approx(0.019882296940110962, rel=1e-1)

def test_overall_prop_efficiency():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import load_aircraft_engine_params
    c_t_available = 0.019882296940110962
    c_t = 0.01946566716046029
    c_t = np.clip(c_t, 0, c_t_available)
    c_t_eta_b = 0.03267886563990706
    eta_over_eta_b_min = 0.5
    aircraft_parameters = load_aircraft_engine_params("A320")
    mach_number = 0.78
    assert atm_params.overall_propulsion_efficiency(mach_number, c_t, c_t_eta_b, aircraft_parameters, eta_over_eta_b_min) == pytest.approx(0.28091506251406534, rel=1e-2)

def test_fuel_mass_flow_rate():
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import load_aircraft_engine_params
    c_t = 0.01946566716046029
    engine_efficiency = 0.28091506251406534
    q_fuel = 43.13e6
    aircraft_parameters = load_aircraft_engine_params("A320")
    wing_surface_area = aircraft_parameters['wing_surface_area']
    T_15 = 288.15
    mach_number = 0.78
    pressure_0 = 100_000
    assert atm_params.fuel_mass_flow_rate(pressure_0, T_15, mach_number, c_t, engine_efficiency, wing_surface_area, q_fuel) == pytest.approx(702.9593810712935, rel=1e-1)

@pytest.mark.parametrize("rocd, expected", [
    (0, 1.8787841814021262),
    (250, 1.8787841814021262),
    (-250, 0.6449999999999999),
    (251, 1.8787841814021262),
    (-251, 0.6449999999999999),
])
def test_fuel_flow_correction(rocd, expected):
    import skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters as atm_params
    from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import load_aircraft_engine_params
    MIN_CRUISE_ALTITUDE = 20_000.0
    rocd_threshold = 250.0
    fuel_flow = 702.9593810712935
    altitude_current = 34_000
    cruise = (rocd < rocd_threshold) & (rocd > -rocd_threshold) & (altitude_current > MIN_CRUISE_ALTITUDE)
    climb = ~cruise & (rocd > 0.0)
    descent = ~cruise & (rocd < 0.0)

    # convert to string 
    flight_phase = np.where(cruise, 'cruise', np.where(climb, 'climb', 'descent'))
    aircraft_parameters = load_aircraft_engine_params("A320")
    T_15 = 288.15
    mach_number = 0.78
    pressure_0 = 100_000
    assert atm_params.fuel_flow_correction(fuel_flow, altitude_current, T_15, pressure_0, mach_number, aircraft_parameters["ff_idle_sls"], aircraft_parameters["ff_max_sls"], flight_phase) == pytest.approx(expected, rel=1e-1)