(window.webpackJsonp=window.webpackJsonp||[]).push([[64],{577:function(a,t,e){"use strict";e.r(t);var r=e(38),s=Object(r.a)({},(function(){var a=this,t=a.$createElement,e=a._self._c||t;return e("ContentSlotsDistributor",{attrs:{"slot-key":a.$parent.slotKey}},[e("h1",{attrs:{id:"hub-domain-flight-planning-aircraft-performance-poll-schumann-utils-parameters-atmospheric-parameters"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#hub-domain-flight-planning-aircraft-performance-poll-schumann-utils-parameters-atmospheric-parameters"}},[a._v("#")]),a._v(" hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters.atmospheric_parameters")]),a._v(" "),e("div",{staticClass:"custom-block tip"},[e("p",{staticClass:"custom-block-title"},[a._v("Domain specification")]),a._v(" "),e("skdecide-summary")],1),a._v(" "),e("h2",{attrs:{id:"reynolds-number"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#reynolds-number"}},[a._v("#")]),a._v(" reynolds_number")]),a._v(" "),e("skdecide-signature",{attrs:{name:"reynolds_number",sig:{params:[{name:"wing_span",annotation:"<class 'float'>"},{name:"wing_aspect_ratio",annotation:"<class 'float'>"},{name:"mach_num",annotation:"<class 'float'>"},{name:"air_temperature",annotation:"<class 'float'>"},{name:"air_pressure",annotation:"<class 'float'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate the Reynolds number.")]),a._v(" "),e("h4",{attrs:{id:"parameters"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("wing_span (float):\nWing surface area, [:math:"),e("code",[a._v("m")]),a._v("]\nwing_aspect_ratio (float):\nWing aspect ratio, [:math:"),e("code",[a._v("-")]),a._v("]\nmach_num (float):\nMach number, [:math:"),e("code",[a._v("-")]),a._v("]\nair_temperature (float):\nAir temperature, [:math:"),e("code",[a._v("K")]),a._v("]\nair_pressure (float):\nAir pressure, [:math:"),e("code",[a._v("Pa")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Reynolds number, [:math:`-`]\n")])])]),e("h2",{attrs:{id:"local-speed-of-sound"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#local-speed-of-sound"}},[a._v("#")]),a._v(" local_speed_of_sound")]),a._v(" "),e("skdecide-signature",{attrs:{name:"local_speed_of_sound",sig:{params:[{name:"air_temperature",annotation:"<class 'float'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate the local speed of sound.")]),a._v(" "),e("h4",{attrs:{id:"parameters-2"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-2"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("air_temperature (float):\nAir temperature, [:math:"),e("code",[a._v("K")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns-2"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-2"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Local speed of sound, [:math:`m/s`]\n")])])]),e("h2",{attrs:{id:"air-density"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#air-density"}},[a._v("#")]),a._v(" air_density")]),a._v(" "),e("skdecide-signature",{attrs:{name:"air_density",sig:{params:[{name:"air_pressure",annotation:"<class 'float'>"},{name:"air_temperature",annotation:"<class 'float'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate the air density.")]),a._v(" "),e("h4",{attrs:{id:"parameters-3"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-3"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("air_pressure (float):\nAir pressure, [:math:"),e("code",[a._v("Pa")]),a._v("]\nair_temperature (float):\nAir temperature, [:math:"),e("code",[a._v("K")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns-3"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-3"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Air density, [:math:`kg/m^3`]\n")])])]),e("h2",{attrs:{id:"dynamic-viscosity"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#dynamic-viscosity"}},[a._v("#")]),a._v(" dynamic_viscosity")]),a._v(" "),e("skdecide-signature",{attrs:{name:"dynamic_viscosity",sig:{params:[{name:"air_temperature",annotation:"<class 'float'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate approximation of the dynamic viscosity.")]),a._v(" "),e("h4",{attrs:{id:"parameters-4"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-4"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("air_temperature (float):\nAir temperature, [:math:"),e("code",[a._v("K")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns-4"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-4"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Dynamic viscosity, [:math:`kg m^{-1} s^{-1}`]\n")])])]),e("h2",{attrs:{id:"lift-coefficient"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#lift-coefficient"}},[a._v("#")]),a._v(" lift_coefficient")]),a._v(" "),e("skdecide-signature",{attrs:{name:"lift_coefficient",sig:{params:[{name:"wing_surface_area",annotation:"<class 'float'>"},{name:"aircraft_mass",annotation:"<class 'float'>"},{name:"air_pressure",annotation:"<class 'float'>"},{name:"air_temperature",annotation:"<class 'float'>"},{name:"mach_num",annotation:"<class 'float'>"},{name:"climb_angle",annotation:"<class 'float'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate the lift coefficient.")]),a._v(" "),e("h4",{attrs:{id:"parameters-5"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-5"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("wing_surface_area (float):\nWing surface area, [:math:"),e("code",[a._v("m^2")]),a._v("]\naircraft_mass (float):\nAircraft mass, [:math:"),e("code",[a._v("kg")]),a._v("]\nair_pressure (float):\nAir pressure, [:math:"),e("code",[a._v("Pa")]),a._v("]\nmach_num (float):\nMach number, [:math:"),e("code",[a._v("-")]),a._v("]\nclimb_angle (float):\nClimb angle, [:math:"),e("code",[a._v("deg")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns-5"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-5"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Lift coefficient, [:math:`-`]\n")])])]),e("h2",{attrs:{id:"skin-friction-coefficient"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#skin-friction-coefficient"}},[a._v("#")]),a._v(" skin_friction_coefficient")]),a._v(" "),e("skdecide-signature",{attrs:{name:"skin_friction_coefficient",sig:{params:[{name:"reynolds_number",annotation:"<class 'float'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate the skin friction coefficient.")]),a._v(" "),e("h4",{attrs:{id:"parameters-6"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-6"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("reynolds_number (float):\nReynolds number, [:math:"),e("code",[a._v("-")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns-6"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-6"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Skin friction coefficient, [:math:`-`]\n")])])]),e("h2",{attrs:{id:"zero-lift-drag-coefficient"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#zero-lift-drag-coefficient"}},[a._v("#")]),a._v(" zero_lift_drag_coefficient")]),a._v(" "),e("skdecide-signature",{attrs:{name:"zero_lift_drag_coefficient",sig:{params:[{name:"c_f",annotation:"<class 'float'>"},{name:"psi_0",annotation:"<class 'float'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate the zero-lift drag coefficient.")]),a._v(" "),e("h4",{attrs:{id:"parameters-7"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-7"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("c_f (float):\nSkin friction coefficient, [:math:"),e("code",[a._v("-")]),a._v("]\npsi_0 (float):\nMiscellaneous drag factor, [:math:"),e("code",[a._v("-")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns-7"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-7"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Zero-lift drag coefficient, [:math:`-`]\n")])])]),e("h2",{attrs:{id:"oswald-efficiency-factor"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#oswald-efficiency-factor"}},[a._v("#")]),a._v(" oswald_efficiency_factor")]),a._v(" "),e("skdecide-signature",{attrs:{name:"oswald_efficiency_factor",sig:{params:[{name:"c_drag_0",annotation:"<class 'float'>"},{name:"aircraft_parameters",annotation:"dict[str, float]"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate the Oswald efficiency factor.")]),a._v(" "),e("h4",{attrs:{id:"parameters-8"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-8"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("c_drag_0 (float):\nZero-lift drag coefficient, [:math:"),e("code",[a._v("-")]),a._v("]\naircraft_parameters (dict[str, float]):\nAircraft parameters.")]),a._v(" "),e("h4",{attrs:{id:"returns-8"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-8"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Oswald efficiency factor, [:math:`-`]\n")])])]),e("h2",{attrs:{id:"non-vortex-lift-dependent-drag-factor"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#non-vortex-lift-dependent-drag-factor"}},[a._v("#")]),a._v(" _non_vortex_lift_dependent_drag_factor")]),a._v(" "),e("skdecide-signature",{attrs:{name:"_non_vortex_lift_dependent_drag_factor",sig:{params:[{name:"c_drag_0",annotation:"<class 'float'>"},{name:"cos_sweep",annotation:"<class 'float'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate the miscellaneous lift-dependent drag factor.")]),a._v(" "),e("h4",{attrs:{id:"parameters-9"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-9"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("c_drag_0 (float):\nZero-lift drag coefficient, [:math:"),e("code",[a._v("-")]),a._v("]\ncos_sweep (float):\nCosine of the wing sweep angle, [:math:"),e("code",[a._v("-")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns-9"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-9"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Miscellaneous lift-dependent drag factor, [:math:`-`]\n")])])]),e("h2",{attrs:{id:"wave-drag-coefficient"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#wave-drag-coefficient"}},[a._v("#")]),a._v(" wave_drag_coefficient")]),a._v(" "),e("skdecide-signature",{attrs:{name:"wave_drag_coefficient",sig:{params:[{name:"mach_num",annotation:"<class 'float'>"},{name:"c_lift",annotation:"<class 'float'>"},{name:"aircraft_parameters",annotation:"dict[str, float]"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate the wave drag coefficient.")]),a._v(" "),e("h4",{attrs:{id:"parameters-10"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-10"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("mach_num (float):\nMach number, [:math:"),e("code",[a._v("-")]),a._v("]\nc_lift (float):\nLift coefficient, [:math:"),e("code",[a._v("-")]),a._v("]\naircraft_parameters (dict[str, float]):\nAircraft parameters.")]),a._v(" "),e("h4",{attrs:{id:"returns-10"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-10"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Wave drag coefficient, [:math:`-`]\n")])])]),e("h2",{attrs:{id:"airframe-drag-coefficient"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#airframe-drag-coefficient"}},[a._v("#")]),a._v(" airframe_drag_coefficient")]),a._v(" "),e("skdecide-signature",{attrs:{name:"airframe_drag_coefficient",sig:{params:[{name:"c_drag_0",annotation:"<class 'float'>"},{name:"c_drag_w",annotation:"<class 'float'>"},{name:"c_lift",annotation:"<class 'float'>"},{name:"e_ls",annotation:"<class 'float'>"},{name:"wing_aspect_ratio",annotation:"<class 'float'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate total airframe drag coefficient.")]),a._v(" "),e("h4",{attrs:{id:"parameters-11"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-11"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("c_drag_0 (float):\nZero-lift drag coefficient, [:math:"),e("code",[a._v("-")]),a._v("]\nc_drag_w (float):\nWave drag coefficient, [:math:"),e("code",[a._v("-")]),a._v("]\nc_lift (float):\nLift coefficient, [:math:"),e("code",[a._v("-")]),a._v("]\ne_ls (float):\nOswald efficiency factor, [:math:"),e("code",[a._v("-")]),a._v("]\nwing_aspect_ratio (float):\nWing aspect ratio, [:math:"),e("code",[a._v("-")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns-11"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-11"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Total airframe drag coefficient, [:math:`-`]\n")])])]),e("h2",{attrs:{id:"thrust-force"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#thrust-force"}},[a._v("#")]),a._v(" thrust_force")]),a._v(" "),e("skdecide-signature",{attrs:{name:"thrust_force",sig:{params:[{name:"aircraft_mass",annotation:"<class 'float'>"},{name:"c_l",annotation:"<class 'float'>"},{name:"c_d",annotation:"<class 'float'>"},{name:"dv_dt",annotation:"<class 'float'>"},{name:"theta",annotation:"<class 'float'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate thrust force summed over all engines.")]),a._v(" "),e("h4",{attrs:{id:"parameters-12"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-12"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("aircraft_mass (float):\nAircraft mass, [:math:"),e("code",[a._v("kg")]),a._v("]\nc_l (float):\nLift coefficient, [:math:"),e("code",[a._v("-")]),a._v("]\nc_d (float):\nDrag coefficient, [:math:"),e("code",[a._v("-")]),a._v("]\ndv_dt (float):\nRate of change of velocity, [:math:"),e("code",[a._v("m/s^2")]),a._v("]\ntheta (float):\nFlight path angle, [:math:"),e("code",[a._v("deg")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns-12"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-12"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Thrust force, [:math:`N`]\n")])])]),e("h2",{attrs:{id:"engine-thrust-coefficient"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#engine-thrust-coefficient"}},[a._v("#")]),a._v(" engine_thrust_coefficient")]),a._v(" "),e("skdecide-signature",{attrs:{name:"engine_thrust_coefficient",sig:{params:[{name:"f_thrust",annotation:"<class 'float'>"},{name:"mach_num",annotation:"<class 'float'>"},{name:"air_pressure",annotation:"<class 'float'>"},{name:"wing_surface_area",annotation:"<class 'float'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate engine thrust coefficient.")]),a._v(" "),e("h4",{attrs:{id:"parameters-13"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-13"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("f_thrust (float):\nThrust force, [:math:"),e("code",[a._v("N")]),a._v("]\nmach_num (float):\nMach number, [:math:"),e("code",[a._v("-")]),a._v("]\nair_pressure (float):\nAir pressure, [:math:"),e("code",[a._v("Pa")]),a._v("]\nwing_surface_area (float):\nWing surface area, [:math:"),e("code",[a._v("m^2")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns-13"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-13"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Engine thrust coefficient, [:math:`-`]\n")])])]),e("h2",{attrs:{id:"overall-propulsion-efficiency"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#overall-propulsion-efficiency"}},[a._v("#")]),a._v(" overall_propulsion_efficiency")]),a._v(" "),e("skdecide-signature",{attrs:{name:"overall_propulsion_efficiency",sig:{params:[{name:"mach_num",annotation:"<class 'float'>"},{name:"c_t",annotation:"<class 'float'>"},{name:"c_t_eta_b",annotation:"<class 'float'>"},{name:"aircraft_parameters",annotation:"dict[str, float]"},{name:"eta_over_eta_b_min",annotation:"<class 'float'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate overall propulsion efficiency.")]),a._v(" "),e("h4",{attrs:{id:"parameters-14"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-14"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("mach_num (float):\nMach number, [:math:"),e("code",[a._v("-")]),a._v("]\nc_t (float):\nThrust coefficient, [:math:"),e("code",[a._v("-")]),a._v("]\nc_t_eta_b (float):\nThrust coefficient at maximum efficiency, [:math:"),e("code",[a._v("-")]),a._v("]\naircraft_parameters (dict[str, float]):\nAircraft parameters.\neta_over_eta_b_min (float):\nMinimum engine efficiency, [:math:"),e("code",[a._v("-")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns-14"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-14"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Overall propulsion efficiency, [:math:`-`]\n")])])]),e("h2",{attrs:{id:"propulsion-efficiency-over-max-propulsion-efficiency"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#propulsion-efficiency-over-max-propulsion-efficiency"}},[a._v("#")]),a._v(" propulsion_efficiency_over_max_propulsion_efficiency")]),a._v(" "),e("skdecide-signature",{attrs:{name:"propulsion_efficiency_over_max_propulsion_efficiency",sig:{params:[{name:"mach_num",annotation:"<class 'float'>"},{name:"c_t",annotation:"<class 'float'>"},{name:"c_t_eta_b",annotation:"<class 'float'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate propulsion efficiency over maximum propulsion efficiency.")]),a._v(" "),e("h4",{attrs:{id:"parameters-15"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-15"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("mach_num (float):\nMach number, [:math:"),e("code",[a._v("-")]),a._v("]\nc_t (float):\nThrust coefficient, [:math:"),e("code",[a._v("-")]),a._v("]\nc_t_eta_b (float):\nThrust coefficient at maximum efficiency, [:math:"),e("code",[a._v("-")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns-15"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-15"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Propulsion efficiency over maximum propulsion efficiency, [:math:`-`]\n")])])]),e("h2",{attrs:{id:"thrust-coefficient-at-max-efficiency"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#thrust-coefficient-at-max-efficiency"}},[a._v("#")]),a._v(" thrust_coefficient_at_max_efficiency")]),a._v(" "),e("skdecide-signature",{attrs:{name:"thrust_coefficient_at_max_efficiency",sig:{params:[{name:"mach_num",annotation:"<class 'float'>"},{name:"m_des",annotation:"<class 'float'>"},{name:"c_t_des",annotation:"<class 'float'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate thrust coefficient at maximum overall propulsion efficiency for a given Mach Number.")]),a._v(" "),e("h4",{attrs:{id:"parameters-16"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-16"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("mach_num (float):\nMach number, [:math:"),e("code",[a._v("-")]),a._v("]\nm_des (float):\nDesign Mach number, [:math:"),e("code",[a._v("-")]),a._v("]\nc_t_des (float):\nThrust coefficient at design Mach number, [:math:"),e("code",[a._v("-")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns-16"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-16"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Thrust coefficient at maximum overall propulsion efficiency, [:math:`-`]\n")])])]),e("h2",{attrs:{id:"max-overall-propulsion-efficiency"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#max-overall-propulsion-efficiency"}},[a._v("#")]),a._v(" max_overall_propulsion_efficiency")]),a._v(" "),e("skdecide-signature",{attrs:{name:"max_overall_propulsion_efficiency",sig:{params:[{name:"mach_num",annotation:"<class 'float'>"},{name:"eta_1",annotation:"<class 'float'>"},{name:"eta_2",annotation:"<class 'float'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate maximum overall propulsion efficiency.")]),a._v(" "),e("h4",{attrs:{id:"parameters-17"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-17"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("mach_num (float):\nMach number, [:math:"),e("code",[a._v("-")]),a._v("]\neta_1 (float):\nEfficiency parameter 1, [:math:"),e("code",[a._v("-")]),a._v("]\neta_2 (float):\nEfficiency parameter 2, [:math:"),e("code",[a._v("-")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns-17"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-17"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Maximum overall propulsion efficiency, [:math:`-`]\n")])])]),e("h2",{attrs:{id:"fuel-mass-flow-rate"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#fuel-mass-flow-rate"}},[a._v("#")]),a._v(" fuel_mass_flow_rate")]),a._v(" "),e("skdecide-signature",{attrs:{name:"fuel_mass_flow_rate",sig:{params:[{name:"air_pressure",annotation:"<class 'float'>"},{name:"air_temperature",annotation:"<class 'float'>"},{name:"mach_num",annotation:"<class 'float'>"},{name:"c_t",annotation:"<class 'float'>"},{name:"eta",annotation:"<class 'float'>"},{name:"wing_surface_area",annotation:"<class 'float'>"},{name:"q_fuel",annotation:"<class 'float'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Calculate fuel mass flow rate.")]),a._v(" "),e("h4",{attrs:{id:"parameters-18"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-18"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("air_pressure (float):\nAir pressure, [:math:"),e("code",[a._v("Pa")]),a._v("]\nair_temperature (float):\nAir temperature, [:math:"),e("code",[a._v("K")]),a._v("]\nmach_num (float):\nMach number, [:math:"),e("code",[a._v("-")]),a._v("]\nc_t (float):\nThrust coefficient, [:math:"),e("code",[a._v("-")]),a._v("]\neta (float):\nEngine efficiency, [:math:"),e("code",[a._v("-")]),a._v("]\nwing_surface_area (float):\nWing surface area, [:math:"),e("code",[a._v("m^2")]),a._v("]\nq_fuel (float):\nFuel heating value, [:math:"),e("code",[a._v("J/kg")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns-18"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-18"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Fuel mass flow rate, [:math:`kg/s`]\n")])])]),e("h2",{attrs:{id:"fuel-flow-correction"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#fuel-flow-correction"}},[a._v("#")]),a._v(" fuel_flow_correction")]),a._v(" "),e("skdecide-signature",{attrs:{name:"fuel_flow_correction",sig:{params:[{name:"fuel_flow",annotation:"<class 'float'>"},{name:"altitude_ft",annotation:"<class 'float'>"},{name:"air_temperature",annotation:"<class 'float'>"},{name:"air_pressure",annotation:"<class 'float'>"},{name:"mach_num",annotation:"<class 'float'>"},{name:"fuel_flow_idle_sls",annotation:"<class 'float'>"},{name:"fuel_flow_max_sls",annotation:"<class 'float'>"},{name:"flight_phase",annotation:"<class 'str'>"}],return:"<class 'float'>"}}}),a._v(" "),e("p",[a._v("Correct fuel flow.")]),a._v(" "),e("h4",{attrs:{id:"parameters-19"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters-19"}},[a._v("#")]),a._v(" Parameters")]),a._v(" "),e("p",[a._v("fuel_flow (float):\nFuel flow, [:math:"),e("code",[a._v("kg/s")]),a._v("]\naltitude_ft (float):\nAltitude, [:math:"),e("code",[a._v("ft")]),a._v("]\nair_temperature (float):\nAir temperature, [:math:"),e("code",[a._v("K")]),a._v("]\nair_pressure (float):\nAir pressure, [:math:"),e("code",[a._v("Pa")]),a._v("]\nmach_num (float):\nMach number, [:math:"),e("code",[a._v("-")]),a._v("]\nfuel_flow_idle_sls (float):\nFuel flow at idle at sea level, [:math:"),e("code",[a._v("kg/s")]),a._v("]\nfuel_flow_max_sls (float):\nMaximum fuel flow at sea level, [:math:"),e("code",[a._v("kg/s")]),a._v("]\nflight_phase (str):\nFlight phase, [:math:"),e("code",[a._v("-")]),a._v("]")]),a._v(" "),e("h4",{attrs:{id:"returns-19"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#returns-19"}},[a._v("#")]),a._v(" Returns")]),a._v(" "),e("div",{staticClass:"language- extra-class"},[e("pre",[e("code",[a._v("float: Corrected fuel flow, [:math:`kg/s`]\n")])])])],1)}),[],!1,null,null,null);t.default=s.exports}}]);