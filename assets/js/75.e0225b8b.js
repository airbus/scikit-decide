(window.webpackJsonp=window.webpackJsonp||[]).push([[75],{589:function(e,t,a){"use strict";a.r(t);var r=a(38),s=Object(r.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("h1",{attrs:{id:"hub-domain-flight-planning-weather-interpolator-weather-tools-parser-pygrib"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#hub-domain-flight-planning-weather-interpolator-weather-tools-parser-pygrib"}},[e._v("#")]),e._v(" hub.domain.flight_planning.weather_interpolator.weather_tools.parser_pygrib")]),e._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[e._v("Domain specification")]),e._v(" "),a("skdecide-summary")],1),e._v(" "),a("h2",{attrs:{id:"computetimestamps"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#computetimestamps"}},[e._v("#")]),e._v(" computeTimeStamps")]),e._v(" "),a("skdecide-signature",{attrs:{name:"computeTimeStamps",sig:{params:[{name:"dates"},{name:"times"},{name:"steps"}]}}}),e._v(" "),a("p",[e._v("This method computes the time stamp from dates, times and step from forecast.")]),e._v(" "),a("p",[e._v(":param dates: List of dates with the following format YYYYMMDD.\n:type dates: list\n:param times: List of forecast time provided as integers.\n:type times: list\n:param steps: List of forecast steps provided as integers.\n:type steps: list\n:return: List of timestamps as datetimes.\n:rtype: list")]),e._v(" "),a("h2",{attrs:{id:"computetimestamp"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#computetimestamp"}},[e._v("#")]),e._v(" computeTimeStamp")]),e._v(" "),a("skdecide-signature",{attrs:{name:"computeTimeStamp",sig:{params:[{name:"date"},{name:"time"},{name:"step"}]}}}),e._v(" "),a("p",[e._v("This method computes a single time stamp value from a forecast date, time and step.")]),e._v(" "),a("p",[e._v(":param date: date with the following format YYYYMMDD.\n:type date: int\n:param time: forecast time value.\n:type time: int\n:param step: forecast step value.\n:type step: int\n:return: timestamp\n:rtype: datetime.datetime")]),e._v(" "),a("h2",{attrs:{id:"flip-matrix"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#flip-matrix"}},[e._v("#")]),e._v(" flip_matrix")]),e._v(" "),a("skdecide-signature",{attrs:{name:"flip_matrix",sig:{params:[{name:"matrix"}]}}}),e._v(" "),a("p",[e._v("Method to flip reverse the order of the elements in the latitude axis of an array.")]),e._v(" "),a("p",[e._v(":param matrix: Dictionary with the matrix to perform the flip.\n:type matrix: dict\n:return: Dictionary with the flipped matrix.\n:rtype: dict")]),e._v(" "),a("h2",{attrs:{id:"gribpygribuniqueforecast"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#gribpygribuniqueforecast"}},[e._v("#")]),e._v(" GribPygribUniqueForecast")]),e._v(" "),a("p",[e._v("This class transforms a grib file into a dictionary containing the values of desired parameters. It allows the\nextraction of just one variable or set of variables corresponding to: CAT, Windshear, Icing, Wind uncertainty and\nconvection. It is based on Pygrib library.")]),e._v(" "),a("h3",{attrs:{id:"constructor"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#constructor"}},[e._v("#")]),e._v(" Constructor "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"GribPygribUniqueForecast",sig:{params:[{name:"grib_path"},{name:"grib_name"},{name:"selected_forecast_dates",default:"None"},{name:"selected_levels",default:"None"}]}}}),e._v(" "),a("p",[e._v("Initialization.")]),e._v(" "),a("p",[e._v(":param grib_path: Grib file path.\n:type grib_path: str\n:param grib_name: Grib file name.\n:type grib_name: str\n:param selected_forecast_dates: List of selected forecast dates.\n:type selected_forecast_dates: list\n:param selected_levels: List of selected levels.\n:type selected_levels: list")]),e._v(" "),a("h3",{attrs:{id:"getcps"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#getcps"}},[e._v("#")]),e._v(" getCPs "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"getCPs",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("This function gets the parameter 'cp': Convective precipitation.\nIt makes use of the method :func:"),a("code",[e._v("getParameters")]),e._v(".")]),e._v(" "),a("p",[e._v(":return: Dictionary with the array containing parameter 'cp'.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"getomega"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#getomega"}},[e._v("#")]),e._v(" getOmega "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"getOmega",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("This function gets the parameter 'w': Vertical velocity (Pa/s).\nIt makes use of the method :func:"),a("code",[e._v("getParameters")]),e._v(".")]),e._v(" "),a("p",[e._v(":return: Dictionary with the array containing parameter 'w'.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"getparameters"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#getparameters"}},[e._v("#")]),e._v(" getParameters "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"getParameters",sig:{params:[{name:"self"},{name:"parameters"},{name:"level",default:"None"}]}}}),e._v(" "),a("p",[e._v("This method returns a dictionary in which each key corresponds to a different parameter from "),a("em",[e._v("parameters")]),e._v(". The\nassociated value is another dictionary with keys: 'values' (array with the parameter values), 'longs'\n(array of longitudes), 'lats' (array of latitudes), 'levels' (array with the pressure levels) and 'times'\n(array of times).")]),e._v(" "),a("p",[e._v(":param parameters: Parameters to get from the grib file.\n:type parameters: list\n:param level: List of pressure levels.\n:type level: list\n:return: Dictionary with arrays corresponding to the values of the different parameters.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"getrelativehumidity"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#getrelativehumidity"}},[e._v("#")]),e._v(" getRelativeHumidity "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"getRelativeHumidity",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("This function gets the parameter 'r': Relative humidity (%).\nIt makes use of the method :func:"),a("code",[e._v("getParameters")]),e._v(".")]),e._v(" "),a("p",[e._v(":return: Dictionary with the array containing parameter 'r'.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"gettts"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#gettts"}},[e._v("#")]),e._v(" getTTs "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"getTTs",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("This function gets the parameter 'totalx': Total totals index (K).\nIt makes use of the method :func:"),a("code",[e._v("getParameters")]),e._v(".")]),e._v(" "),a("p",[e._v(":return: Dictionary with the array containing parameter 'totalx'.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"gettemps"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#gettemps"}},[e._v("#")]),e._v(" getTemps "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"getTemps",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("This function gets the parameter 't': Temperature (K).\nIt makes use of the method :func:"),a("code",[e._v("getParameters")]),e._v(".")]),e._v(" "),a("p",[e._v(":return: Dictionary with the array containing parameter 't'.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"getus"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#getus"}},[e._v("#")]),e._v(" getUs "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"getUs",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("This function gets the parameter 'u': U component of wind (m/s).\nIt makes use of the method :func:"),a("code",[e._v("getParameters")]),e._v(".")]),e._v(" "),a("p",[e._v(":return: Dictionary with the array containing parameter 'u'.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"getvs"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#getvs"}},[e._v("#")]),e._v(" getVs "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"getVs",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("This function gets the parameter 'v': V component of wind (m/s).\nIt makes use of the method :func:"),a("code",[e._v("getParameters")]),e._v(".")]),e._v(" "),a("p",[e._v(":return: Dictionary with the array containing parameter 'v'.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"get-grib-all-parameters-donuts-pl"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-grib-all-parameters-donuts-pl"}},[e._v("#")]),e._v(" get_grib_all_parameters_donuts_pl "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_grib_all_parameters_donuts_pl",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("This method gets the parameters used in DONUT'S PL calculations : 'u' (U component of wind), 'v'\n(V component of wind), 'z' (Geopotential), 't' (Temperature), 'r' (Relative humidity)\nand 'w' (Vertical velocity) . It makes use of the method :func:"),a("code",[e._v("getParameters")]),e._v(".")]),e._v(" "),a("p",[e._v(":return: Dictionary containing values of parameters 'u', 'v', 'z', 't', 'r' and 'w'.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"get-grib-all-parameters-donuts-sfc"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-grib-all-parameters-donuts-sfc"}},[e._v("#")]),e._v(" get_grib_all_parameters_donuts_sfc "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_grib_all_parameters_donuts_sfc",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("This method gets the parameters used in DONUT'S SFC calculations : 'totalx' (totals total index) and 'cp'\n(convective precipitation). It makes use of the method :func:"),a("code",[e._v("getParameters")]),e._v(".")]),e._v(" "),a("p",[e._v(":return:  Dictionary containing values of parameters 'totalx' and 'cp'.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"get-grib-all-parameters-pl"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-grib-all-parameters-pl"}},[e._v("#")]),e._v(" get_grib_all_parameters_pl "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_grib_all_parameters_pl",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("This method gets all the parameters used in PL calculations. It makes use of the method :func:"),a("code",[e._v("getParameters")]),e._v(".")]),e._v(" "),a("p",[e._v(":return: Dictionary containing values of all the parameters.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"get-grib-all-parameters-sfc"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-grib-all-parameters-sfc"}},[e._v("#")]),e._v(" get_grib_all_parameters_sfc "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_grib_all_parameters_sfc",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("This method gets all the parameters used in SFC calculations. It makes use of the method :func:"),a("code",[e._v("getParameters")]),e._v(".")]),e._v(" "),a("p",[e._v(":return: Dictionary containing values of all the parameters.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"get-grib-cat-xarray"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-grib-cat-xarray"}},[e._v("#")]),e._v(" get_grib_cat_xarray "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_grib_cat_xarray",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("This method gets the parameters used in CAT calculations: 'u' (U component of wind), 'v' (V component of wind)\nand 'z' (Geopotential). It makes use of the method :func:"),a("code",[e._v("getParameters")]),e._v(".")]),e._v(" "),a("p",[e._v(":return: Dictionary containing values of parameters 'u', 'v' and 'z'.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"get-grib-convection-xarray"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-grib-convection-xarray"}},[e._v("#")]),e._v(" get_grib_convection_xarray "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_grib_convection_xarray",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("This method gets the parameters used in Convection calculations: 'totalx' (Total totals index), 'cp'\n(convective precipitation). It makes use of the method :func:"),a("code",[e._v("getParameters")]),e._v(".")]),e._v(" "),a("p",[e._v(":return: Dictionary containing values of parameters 'totalx' and 'cp'.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"get-grib-icing-xarray"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-grib-icing-xarray"}},[e._v("#")]),e._v(" get_grib_icing_xarray "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_grib_icing_xarray",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("This method gets the parameters used in Icing calculations: 't' (Temperature), 'r' (Relative humidity)\nand 'w' (Vertical velocity). It makes use of the method :func:"),a("code",[e._v("getParameters")]),e._v(".")]),e._v(" "),a("p",[e._v(":return: Dictionary containing values of parameters 't', 'r' and 'w'.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"get-grib-wind-uncertainty-xarray"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-grib-wind-uncertainty-xarray"}},[e._v("#")]),e._v(" get_grib_wind_uncertainty_xarray "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_grib_wind_uncertainty_xarray",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("This method gets the parameters used in Wind Uncertainty calculations: 'u' (U component of wind), 'v' (\nV component of wind) and 't' (Temperature). It makes use of the method :func:"),a("code",[e._v("getParameters")]),e._v(".")]),e._v(" "),a("p",[e._v(":return: Dictionary containing values of parameters 'u', 'v' and 't'.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"get-grib-windshear-xarray"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-grib-windshear-xarray"}},[e._v("#")]),e._v(" get_grib_windshear_xarray "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_grib_windshear_xarray",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("This method gets the parameters used in Windshear calculations: 'u' (U component of wind), 'v' (V component of wind)\nand 'z' (Geopotential). It makes use of the method :func:"),a("code",[e._v("getParameters")]),e._v(".")]),e._v(" "),a("p",[e._v(":return: Dictionary containing values of parameters 'u', 'v' and 'z'.\n:rtype: dict")]),e._v(" "),a("h3",{attrs:{id:"isparameteringrib"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#isparameteringrib"}},[e._v("#")]),e._v(" isParameterInGrib "),a("Badge",{attrs:{text:"GribPygribUniqueForecast",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"isParameterInGrib",sig:{params:[{name:"self"},{name:"parameter"}]}}}),e._v(" "),a("p",[e._v("Method to check if a parameter is in the grib file.")]),e._v(" "),a("p",[e._v(":param parameter: Parameter that needs to be checked.\n:type parameter: str\n:return: True if is in grib, False otherwise.\n:rtype: bool")])],1)}),[],!1,null,null,null);t.default=s.exports}}]);