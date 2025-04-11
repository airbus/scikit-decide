(window.webpackJsonp=window.webpackJsonp||[]).push([[80],{595:function(e,t,n){"use strict";n.r(t);var o=n(38),a=Object(o.a)({},(function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[n("h1",{attrs:{id:"hub-domain-flight-planning-weather-interpolator-weather-tools-unit-conversion"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#hub-domain-flight-planning-weather-interpolator-weather-tools-unit-conversion"}},[e._v("#")]),e._v(" hub.domain.flight_planning.weather_interpolator.weather_tools.unit_conversion")]),e._v(" "),n("div",{staticClass:"custom-block tip"},[n("p",{staticClass:"custom-block-title"},[e._v("Domain specification")]),e._v(" "),n("skdecide-summary")],1),e._v(" "),n("h2",{attrs:{id:"area-conv"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#area-conv"}},[e._v("#")]),e._v(" area_conv")]),e._v(" "),n("skdecide-signature",{attrs:{name:"area_conv",sig:{params:[{name:"A"},{name:"from_units",default:"ft**2"},{name:"to_units",default:"ft**2"}]}}}),e._v(" "),n("p",[e._v("Convert area values between ft"),n("strong",[e._v("2, in")]),e._v("2, m"),n("strong",[e._v("2, km")]),e._v("2, sm"),n("strong",[e._v("2 and nm")]),e._v("2.")]),e._v(" "),n("p",[e._v("The incoming value is first converted to ft**2, then it is converted to\ndesired return value.")]),e._v(" "),n("p",[e._v("The units default to those specified in default_units.py")]),e._v(" "),n("p",[e._v("Examples:")]),e._v(" "),n("p",[e._v("Convert 1 ft"),n("strong",[e._v("2 to inches")]),e._v("2, with ft**2 already defined as the default\nunits:")]),e._v(" "),n("blockquote",[n("blockquote",[n("blockquote",[n("p",[e._v("area_conv(1, to_units = 'in**2')\n144.0")])])])]),e._v(" "),n("p",[e._v("Convert 288 square inches to square feet, with ft**2 already defined as the default\nunits:")]),e._v(" "),n("blockquote",[n("blockquote",[n("blockquote",[n("p",[e._v("area_conv(288, from_units = 'in**2')\n2.0")])])])]),e._v(" "),n("p",[e._v("Convert 10 square metres to square inches:")]),e._v(" "),n("blockquote",[n("blockquote",[n("blockquote",[n("p",[e._v("area_conv(1000, from_units = 'm"),n("strong",[e._v("2', to_units = 'in")]),e._v("2')\n1550003.1000061999")])])])]),e._v(" "),n("h2",{attrs:{id:"density-conv"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#density-conv"}},[e._v("#")]),e._v(" density_conv")]),e._v(" "),n("skdecide-signature",{attrs:{name:"density_conv",sig:{params:[{name:"D"},{name:"from_units"},{name:"to_units"}]}}}),e._v(" "),n("p",[e._v("Convert density values between kg/m"),n("strong",[e._v("3, slug/ft")]),e._v("3 and lb/ft**3.")]),e._v(" "),n("p",[e._v("The incoming value is first converted to kg/m**3, then it is converted\nto desired return value.")]),e._v(" "),n("p",[e._v("There are no default units. Both the from_units and the to_units must\nbe specified.")]),e._v(" "),n("p",[e._v("Example:")]),e._v(" "),n("p",[e._v("Convert 1.225 kg per metre cubed to lb per foot cubed:")]),e._v(" "),n("blockquote",[n("blockquote",[n("blockquote",[n("p",[e._v("density_conv(1.225, from_units = 'kg/m"),n("strong",[e._v("3', to_units = 'lb/ft")]),e._v("3')\n0.076474253491112101")])])])]),e._v(" "),n("h2",{attrs:{id:"force-conv"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#force-conv"}},[e._v("#")]),e._v(" force_conv")]),e._v(" "),n("skdecide-signature",{attrs:{name:"force_conv",sig:{params:[{name:"F"},{name:"from_units",default:"lb"},{name:"to_units",default:"lb"}]}}}),e._v(" "),n("p",[e._v("Convert force values between lb and N.")]),e._v(" "),n("p",[e._v("The incoming value is first converted to N, then it is converted to the\ndesired return value.")]),e._v(" "),n("h2",{attrs:{id:"len-conv"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#len-conv"}},[e._v("#")]),e._v(" len_conv")]),e._v(" "),n("skdecide-signature",{attrs:{name:"len_conv",sig:{params:[{name:"L"},{name:"from_units",default:"ft"},{name:"to_units",default:"ft"}]}}}),e._v(" "),n("p",[e._v("Convert length values between ft, in, m, km, sm and nm.")]),e._v(" "),n("p",[e._v("The incoming value is first converted to ft, then it is converted to\ndesired return value.")]),e._v(" "),n("p",[e._v("The units default to those specified in default_units.py")]),e._v(" "),n("p",[e._v("Examples:")]),e._v(" "),n("p",[e._v("Convert 5280 ft to statute miles, with feet already defined as the default\nunits:")]),e._v(" "),n("blockquote",[n("blockquote",[n("blockquote",[n("p",[e._v("len_conv(5280, to_units = 'sm')\n1.0")])])])]),e._v(" "),n("p",[e._v("Convert 1 nautical mile to feet, with feet already defined as the default\nunits:")]),e._v(" "),n("blockquote",[n("blockquote",[n("blockquote",[n("p",[e._v("len_conv(1, from_units = 'nm')\n6076.1154855643044")])])])]),e._v(" "),n("p",[e._v("Convert 1000 metres to kilometres:")]),e._v(" "),n("blockquote",[n("blockquote",[n("blockquote",[n("p",[e._v("len_conv(1000, from_units = 'm', to_units = 'km')\n0.99999999999999989")])])])]),e._v(" "),n("h2",{attrs:{id:"power-conv"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#power-conv"}},[e._v("#")]),e._v(" power_conv")]),e._v(" "),n("skdecide-signature",{attrs:{name:"power_conv",sig:{params:[{name:"P"},{name:"from_units",default:"hp"},{name:"to_units",default:"hp"}]}}}),e._v(" "),n("p",[e._v("Convert power values between horsepower, ft-lb/mn,  ft-lb/s, watts,\nkilowatts, BTU/hr and BTU/mn.")]),e._v(" "),n("p",[e._v("The incoming value is first converted to hp, then it is converted to the\ndesired return value.\nThe units default to those specified in default_units.py")]),e._v(" "),n("h2",{attrs:{id:"speed-conv"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#speed-conv"}},[e._v("#")]),e._v(" speed_conv")]),e._v(" "),n("skdecide-signature",{attrs:{name:"speed_conv",sig:{params:[{name:"S"},{name:"from_units",default:"kt"},{name:"to_units",default:"kt"}]}}}),e._v(" "),n("p",[e._v("Convert speed values between kt, mph, km/h, m/s and ft/s.")]),e._v(" "),n("p",[e._v("The incoming value is first converted to kt, then it is converted to\ndesired return value.\nThe units default to those specified in default_units.py")]),e._v(" "),n("p",[e._v("Example:")]),e._v(" "),n("p",[e._v("Convert 230 mph  to kt:")]),e._v(" "),n("blockquote",[n("blockquote",[n("blockquote",[n("p",[e._v("speed_conv(230, from_units = 'mph', to_units = 'kt')\n199.86453563714903")])])])]),e._v(" "),n("h2",{attrs:{id:"temp-conv"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#temp-conv"}},[e._v("#")]),e._v(" temp_conv")]),e._v(" "),n("skdecide-signature",{attrs:{name:"temp_conv",sig:{params:[{name:"T"},{name:"from_units",default:"C"},{name:"to_units",default:"C"}]}}}),e._v(" "),n("p",[e._v("Convert absolute temperature values between deg C, F, K and R.")]),e._v(" "),n("p",[e._v("This function should not be used for relative temperature conversions,\ni.e. temperature differences.")]),e._v(" "),n("p",[e._v("The incoming value is first converted to deg K, then it is converted to\ndesired return value.\nThe units default to those specified in default_units.py")]),e._v(" "),n("p",[e._v("Examples:")]),e._v(" "),n("p",[e._v("Convert 32 deg F to deg C, with deg C as the default units:")]),e._v(" "),n("blockquote",[n("blockquote",[n("blockquote",[n("p",[e._v("temp_conv(32, from_units = 'F')\n0.0")])])])]),e._v(" "),n("p",[e._v("Convert 100 deg C to deg F, with deg C as the default units:")]),e._v(" "),n("blockquote",[n("blockquote",[n("blockquote",[n("p",[e._v("temp_conv(100, to_units = 'F')\n212.0")])])])]),e._v(" "),n("p",[e._v("Convert 59 deg F to deg K")]),e._v(" "),n("blockquote",[n("blockquote",[n("blockquote",[n("p",[e._v("temp_conv(59, from_units = 'F', to_units = 'K')\n288.14999999999998")])])])]),e._v(" "),n("h2",{attrs:{id:"vol-conv"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#vol-conv"}},[e._v("#")]),e._v(" vol_conv")]),e._v(" "),n("skdecide-signature",{attrs:{name:"vol_conv",sig:{params:[{name:"V"},{name:"from_units",default:"ft**3"},{name:"to_units",default:"ft**3"}]}}}),e._v(" "),n("p",[e._v("Convert volume values between USG, ImpGal (Imperial gallons), l (litres), ft"),n("strong",[e._v("3, in")]),e._v("3, m"),n("strong",[e._v("3, km")]),e._v("3, sm"),n("strong",[e._v("3 and nm")]),e._v("3.")]),e._v(" "),n("p",[e._v("The incoming value is first converted to ft**3, then it is converted to\ndesired return value.")]),e._v(" "),n("p",[e._v("The units default to those specified in default_units.py")]),e._v(" "),n("p",[e._v("Examples:")]),e._v(" "),n("p",[e._v("Convert 1 cubic foot to US gallons, with cubic feet already defined as\nthe default units:")]),e._v(" "),n("blockquote",[n("blockquote",[n("blockquote",[n("p",[e._v("vol_conv(1, to_units = 'USG')\n7.4805194804946105")])])])]),e._v(" "),n("p",[e._v("Convert 1 Imperial gallon to cubic feet, with cubic feet already defined\nas the default units:")]),e._v(" "),n("blockquote",[n("blockquote",[n("blockquote",[n("p",[e._v("vol_conv(1, from_units = 'ImpGal')\n0.16054365323600001")])])])]),e._v(" "),n("p",[e._v("Convert 10 US gallon to litres:")]),e._v(" "),n("blockquote",[n("blockquote",[n("blockquote",[n("p",[e._v("vol_conv(10, from_units = 'USG', to_units = 'l')\n37.854117840125852")])])])]),e._v(" "),n("h2",{attrs:{id:"wt-conv"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#wt-conv"}},[e._v("#")]),e._v(" wt_conv")]),e._v(" "),n("skdecide-signature",{attrs:{name:"wt_conv",sig:{params:[{name:"W"},{name:"from_units",default:"lb"},{name:"to_units",default:"lb"}]}}}),e._v(" "),n("p",[e._v("Convert weight values between lb and kg.")]),e._v(" "),n("p",[e._v("Purists will yell that lb is a unit of weight, and kg is a unit of mass.\nGet over it.")]),e._v(" "),n("p",[e._v("The incoming value is first converted to kg, then it is converted to the\ndesired return value.")]),e._v(" "),n("p",[e._v("The units default to those specified in default_units.py")]),e._v(" "),n("h2",{attrs:{id:"avgas-conv"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#avgas-conv"}},[e._v("#")]),e._v(" avgas_conv")]),e._v(" "),n("skdecide-signature",{attrs:{name:"avgas_conv",sig:{params:[{name:"AG"},{name:"from_units",default:"lb"},{name:"to_units",default:"lb"},{name:"temp",default:"15"},{name:"temp_units",default:"C"},{name:"grade",default:"nominal"}]}}}),e._v(" "),n("p",[e._v("Convert aviation gasoline between units of lb, US Gallon (USG),\nImperial Gallon (Imp Gal), litres (l) and kg, assuming nominal\ndensity for aviation gasoline of 6.01 lb per USG.")]),e._v(" "),n("p",[e._v("The units default to those specified in default_units.py")]),e._v(" "),n("p",[e._v("Note: it was difficult to find authoritative values for aviation gasoline\ndensity.  Conventional wisdom is that aviation gasoline has a density of\n6 lb/USG.  The Canada Flight Supplement provides densities of:\ntemp      density     density    density\n(deg C)   (lb/USG)  (lb/ImpGal)  (lb/l)\n-40         6.41       7.68       1.69\n-20         6.26       7.50       1.65\n0         6.12       7.33       1.62\n15         6.01       7.20       1.59\n30         5.90       7.07       1.56")]),e._v(" "),n("p",[e._v("However, the Canada Flight Supplement does not provide a source for its\ndensity data.  And, the values for the different volume units are not\ncompletly consistent, as they don't vary by exactly the correct factor.\nFor example, if the density at 15 deg C is 6.01 lb/USG, we would expect\nthe density in lb/ImpGal to be 7.22, (given that 1 ImpGal = 1.201 USG)\nyet the Canada Flight Supplement has 7.20.")]),e._v(" "),n("p",[e._v('The only authoritative source for aviation gasoline density that was\nfound on the web was the "Air BP Handbook of Products" on the British\nPetroleum (BP) web site:')]),e._v(" "),n("p",[e._v("<http://www.bp.com/liveassets/bp_internet/aviation/air_bp/STAGING/local_assets/downloads_pdfs/a/air_bp_products_handbook_04004_1.pdf>")]),e._v(" "),n("p",[e._v("It provides the following density data valid at 15 deg C (the BP document\nonly provides density in kg/m"),n("strong",[e._v("3 - the density in lb/USG were calculated\nby Kevin Horton):\nAvgas    density     density\nType    (kg/m")]),e._v("3)    (lb/USG)\n80       690          5.76\n100      695          5.80\n100LL    715          5.97")]),e._v(" "),n("p",[e._v("The available aviation gasoline specifications do not appear to define an\nallowable density range.  They do define allowable ranges for various\nparametres of the distillation process - the density of the final product\nwill vary depending on where in the allowable range the refinery is run.\nThus there will be some variation in density from refinery to refinery.")]),e._v(" "),n("p",[e._v("This function uses the 15 deg C density values provided by BP, with the\nvariation with temperature provided in the Canada Flight Supplement.")]),e._v(" "),n("p",[e._v('The grade may be specified as "80", "100" or "100LL".  It defaults to\n"100LL" if it is not specified.')]),e._v(" "),n("p",[e._v("The temperature defaults to 15 deg C if it is not specified.")])],1)}),[],!1,null,null,null);t.default=a.exports}}]);