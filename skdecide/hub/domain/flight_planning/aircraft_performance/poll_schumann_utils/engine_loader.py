from pathlib import Path

import pandas as pd

from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters import (
    aircraft_parameters as arc_params,
)

PS_FILE_PATH = str(Path(__file__).parent / "data" / "aircraft_engine_params.csv")


def load_aircraft_engine_params(actype: str):
    """Extract aircraft-engine parameters for each aircraft type supported by the PS model."""
    dtypes = {
        "ICAO": object,
        "Manufacturer": object,
        "Type": object,
        "Year_of_first_flight": float,
        "n_engine": int,
        "winglets": object,
        "WV": object,
        "MTOM_kg": float,
        "MLM_kg": float,
        "MZFM_kg": float,
        "OEM_i_kg": float,
        "MPM_i_kg": float,
        "MZFM_MTOM": float,
        "OEM_i_MTOM": float,
        "MPM_i_MTOM": float,
        "Sref_m2": float,
        "span_m": float,
        "bf_m": float,
        "delta_2": float,
        "cos_sweep": float,
        "AR": float,
        "psi_0": float,
        "Xo": float,
        "wing_constant": float,
        "j_2": float,
        "j_1": float,
        "CL_do": float,
        "nominal_F00_ISA_kn": float,
        "mf_max_T_O_SLS_kg_s": float,
        "mf_idle_SLS_kg_s": float,
        "M_des": float,
        "CT_des": float,
        "eta_1": float,
        "eta_2": float,
        "Mec": float,
        "Tec": float,
        "FL_max": float,
        "MMO": float,
    }

    df = pd.read_csv(PS_FILE_PATH, dtype=dtypes)

    columns_renamed = {
        "Manufacturer": "manufacturer",
        "Type": "aircraft_type",
        "MTOM_kg": "amass_mtow",
        "MLM_kg": "amass_mlw",
        "MZFM_kg": "amass_mzfw",
        "OEM_i_kg": "amass_oew",
        "MPM_i_kg": "amass_mpl",
        "AR": "wing_aspect_ratio",
        "Sref_m2": "wing_surface_area",
        "span_m": "wing_span",
        "bf_m": "fuselage_width",
        "Xo": "x_ref",
        "CL_do": "c_l_do",
        "nominal_F00_ISA_kn": "f_00",
        "mf_max_T_O_SLS_kg_s": "ff_max_sls",
        "mf_idle_SLS_kg_s": "ff_idle_sls",
        "M_des": "m_des",
        "CT_des": "c_t_des",
        "Tec": "tr_ec",
        "Mec": "m_ec",
        "FL_max": "fl_max",
        "MMO": "max_mach_num",
    }

    df.rename(columns=columns_renamed, inplace=True)

    df = df.loc[df["ICAO"] == actype, :]

    df["j_3"] = 70.0
    df["f_00"] = df["f_00"] * 1_000.0
    df["tet_mto"] = arc_params.turbine_entry_temperature_at_max_take_off(
        df["Year_of_first_flight"].values
    )
    df["p_i_max"] = arc_params.impact_pressure_max_operating_limits(
        df["max_mach_num"].values
    )
    df["tet_mcc"] = arc_params.turbine_entry_temperature_at_max_continuous_climb(
        df["tet_mto"].values
    )
    df["p_inf_co"] = arc_params.crossover_pressure_altitude(
        df["max_mach_num"].values, df["p_i_max"].values
    )

    return df.to_dict(orient="records")[0]
