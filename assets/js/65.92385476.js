(window.webpackJsonp=window.webpackJsonp||[]).push([[65],{578:function(e,t,a){"use strict";a.r(t);var r=a(38),s=Object(r.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("h1",{attrs:{id:"hub-domain-flight-planning-aircraft-performance-weather-service-isa-atmosphere-service"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#hub-domain-flight-planning-aircraft-performance-weather-service-isa-atmosphere-service"}},[e._v("#")]),e._v(" hub.domain.flight_planning.aircraft_performance.weather.service.isa_atmosphere_service")]),e._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[e._v("Domain specification")]),e._v(" "),a("skdecide-summary")],1),e._v(" "),a("h2",{attrs:{id:"isaatmosphereservice"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#isaatmosphereservice"}},[e._v("#")]),e._v(" ISAAtmosphereService")]),e._v(" "),a("p",[e._v("Implementation of the ISA atmosphere")]),e._v(" "),a("h3",{attrs:{id:"atmosphere-model-type"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#atmosphere-model-type"}},[e._v("#")]),e._v(" atmosphere_model_type "),a("Badge",{attrs:{text:"AtmopshereServiceInterface",type:"warn"}})],1),e._v(" "),a("h3",{attrs:{id:"retrieve-weather-state"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#retrieve-weather-state"}},[e._v("#")]),e._v(" retrieve_weather_state "),a("Badge",{attrs:{text:"AtmopshereServiceInterface",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"retrieve_weather_state",sig:{params:[{name:"self"},{name:"atmosphere_settings",annotation:"<class 'skdecide.hub.domain.flight_planning.aircraft_performance.weather.settings.isa_atmosphere_settings.IsaAtmosphereSettings'>"},{name:"four_dimensions_state",annotation:"<class 'skdecide.hub.domain.flight_planning.aircraft_performance.bean.four_dimensions_state.FourDimensionsState'>"}],return:"<class 'skdecide.hub.domain.flight_planning.aircraft_performance.bean.weather_state.WeatherState'>"}}}),e._v(" "),a("p",[e._v("From the aircraft state location and atmosphere settings, compute the weather state using the appropriate atmosphere service\n:param atmosphere_settings: Settings defining the atmosphere (type, constants...)\n:param four_dimensions_state: 4D state (zp, location, time)\n:return: Weather state (Temperature, pressure...)")]),e._v(" "),a("h3",{attrs:{id:"isaatmosphereservice-isa-temperature-k"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#isaatmosphereservice-isa-temperature-k"}},[e._v("#")]),e._v(" _ISAAtmosphereService__isa_temperature_k "),a("Badge",{attrs:{text:"ISAAtmosphereService",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_ISAAtmosphereService__isa_temperature_k",sig:{params:[{name:"self"},{name:"altitude_ft",annotation:"<class 'float'>"},{name:"tropo_ft",default:"None",annotation:"typing.Optional[float]"}],return:"<class 'float'>"}}}),e._v(" "),a("p",[e._v("Compute the temperature at the given altitude using a custom value for the tropopause altitude\nArgs:\naltitude_ft (float): user altitude at which the temperature will be computed\ntropo_ft (float): user custom tropopause altitude\nReturns:\nfloat: isa temperature (K)")]),e._v(" "),a("h3",{attrs:{id:"isaatmosphereservice-pressure"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#isaatmosphereservice-pressure"}},[e._v("#")]),e._v(" _ISAAtmosphereService__pressure "),a("Badge",{attrs:{text:"ISAAtmosphereService",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_ISAAtmosphereService__pressure",sig:{params:[{name:"self"},{name:"altitude_ft",default:"0.0"}]}}}),e._v(" "),a("p",[e._v("Compute Static Pressure in Pa from altitude in ft\nRemark: altitude is optional and can be provided as a float,\na numpy or a pd.series\n:param altitude_ft: altitude (ft) (default : 0)\n:return: pressure (Pa)")])],1)}),[],!1,null,null,null);t.default=s.exports}}]);