(window.webpackJsonp=window.webpackJsonp||[]).push([[40],{556:function(e,t,a){"use strict";a.r(t);var r=a(38),s=Object(r.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("h1",{attrs:{id:"builders-domain-scheduling-resource-availability"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#builders-domain-scheduling-resource-availability"}},[e._v("#")]),e._v(" builders.domain.scheduling.resource_availability")]),e._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[e._v("Domain specification")]),e._v(" "),a("skdecide-summary")],1),e._v(" "),a("h2",{attrs:{id:"uncertainresourceavailabilitychanges"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#uncertainresourceavailabilitychanges"}},[e._v("#")]),e._v(" UncertainResourceAvailabilityChanges")]),e._v(" "),a("p",[e._v("A domain must inherit this class if the availability of its resource vary in an uncertain way over time.")]),e._v(" "),a("h3",{attrs:{id:"check-unique-resource-names"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#check-unique-resource-names"}},[e._v("#")]),e._v(" check_unique_resource_names "),a("Badge",{attrs:{text:"UncertainResourceAvailabilityChanges",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"check_unique_resource_names",sig:{params:[{name:"self"}],return:"bool"}}}),e._v(" "),a("p",[e._v("Return True if there are no duplicates in resource names across both resource types\nand resource units name lists.")]),e._v(" "),a("h3",{attrs:{id:"sample-quantity-resource"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-quantity-resource"}},[e._v("#")]),e._v(" sample_quantity_resource "),a("Badge",{attrs:{text:"UncertainResourceAvailabilityChanges",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"sample_quantity_resource",sig:{params:[{name:"self"},{name:"resource",annotation:"str"},{name:"time",annotation:"int"},{name:"**kwargs"}],return:"int"}}}),e._v(" "),a("p",[e._v("Sample an amount of resource availability (int) for the given resource\n(either resource type or resource unit) at the given time. This number should be the sum of the number of\nresource available at time t and the number of resource of this type consumed so far).")]),e._v(" "),a("h3",{attrs:{id:"sample-quantity-resource-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-quantity-resource-2"}},[e._v("#")]),e._v(" _sample_quantity_resource "),a("Badge",{attrs:{text:"UncertainResourceAvailabilityChanges",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_sample_quantity_resource",sig:{params:[{name:"self"},{name:"resource",annotation:"str"},{name:"time",annotation:"int"},{name:"**kwargs"}],return:"int"}}}),e._v(" "),a("p",[e._v("Sample an amount of resource availability (int) for the given resource\n(either resource type or resource unit) at the given time. This number should be the sum of the number of\nresource available at time t and the number of resource of this type consumed so far).")]),e._v(" "),a("h2",{attrs:{id:"deterministicresourceavailabilitychanges"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#deterministicresourceavailabilitychanges"}},[e._v("#")]),e._v(" DeterministicResourceAvailabilityChanges")]),e._v(" "),a("p",[e._v("A domain must inherit this class if the availability of its resource vary in a deterministic way over time.")]),e._v(" "),a("h3",{attrs:{id:"check-unique-resource-names-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#check-unique-resource-names-2"}},[e._v("#")]),e._v(" check_unique_resource_names "),a("Badge",{attrs:{text:"UncertainResourceAvailabilityChanges",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"check_unique_resource_names",sig:{params:[{name:"self"}],return:"bool"}}}),e._v(" "),a("p",[e._v("Return True if there are no duplicates in resource names across both resource types\nand resource units name lists.")]),e._v(" "),a("h3",{attrs:{id:"get-quantity-resource"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-quantity-resource"}},[e._v("#")]),e._v(" get_quantity_resource "),a("Badge",{attrs:{text:"DeterministicResourceAvailabilityChanges",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_quantity_resource",sig:{params:[{name:"self"},{name:"resource",annotation:"str"},{name:"time",annotation:"int"},{name:"**kwargs"}],return:"int"}}}),e._v(" "),a("p",[e._v("Return the resource availability (int) for the given resource\n(either resource type or resource unit) at the given time.")]),e._v(" "),a("h3",{attrs:{id:"sample-quantity-resource-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-quantity-resource-3"}},[e._v("#")]),e._v(" sample_quantity_resource "),a("Badge",{attrs:{text:"UncertainResourceAvailabilityChanges",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"sample_quantity_resource",sig:{params:[{name:"self"},{name:"resource",annotation:"str"},{name:"time",annotation:"int"},{name:"**kwargs"}],return:"int"}}}),e._v(" "),a("p",[e._v("Sample an amount of resource availability (int) for the given resource\n(either resource type or resource unit) at the given time. This number should be the sum of the number of\nresource available at time t and the number of resource of this type consumed so far).")]),e._v(" "),a("h3",{attrs:{id:"get-quantity-resource-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-quantity-resource-2"}},[e._v("#")]),e._v(" _get_quantity_resource "),a("Badge",{attrs:{text:"DeterministicResourceAvailabilityChanges",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_get_quantity_resource",sig:{params:[{name:"self"},{name:"resource",annotation:"str"},{name:"time",annotation:"int"},{name:"**kwargs"}],return:"int"}}}),e._v(" "),a("p",[e._v("Return the resource availability (int) for the given resource\n(either resource type or resource unit) at the given time.")]),e._v(" "),a("h3",{attrs:{id:"sample-quantity-resource-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-quantity-resource-4"}},[e._v("#")]),e._v(" _sample_quantity_resource "),a("Badge",{attrs:{text:"UncertainResourceAvailabilityChanges",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_sample_quantity_resource",sig:{params:[{name:"self"},{name:"resource",annotation:"str"},{name:"time",annotation:"int"},{name:"**kwargs"}],return:"int"}}}),e._v(" "),a("p",[e._v("Sample an amount of resource availability (int) for the given resource\n(either resource type or resource unit) at the given time. This number should be the sum of the number of\nresource available at time t and the number of resource of this type consumed so far).")]),e._v(" "),a("h2",{attrs:{id:"withoutresourceavailabilitychange"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#withoutresourceavailabilitychange"}},[e._v("#")]),e._v(" WithoutResourceAvailabilityChange")]),e._v(" "),a("p",[e._v("A domain must inherit this class if the availability of its resource does not vary over time.")]),e._v(" "),a("h3",{attrs:{id:"check-unique-resource-names-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#check-unique-resource-names-3"}},[e._v("#")]),e._v(" check_unique_resource_names "),a("Badge",{attrs:{text:"UncertainResourceAvailabilityChanges",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"check_unique_resource_names",sig:{params:[{name:"self"}],return:"bool"}}}),e._v(" "),a("p",[e._v("Return True if there are no duplicates in resource names across both resource types\nand resource units name lists.")]),e._v(" "),a("h3",{attrs:{id:"get-original-quantity-resource"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-original-quantity-resource"}},[e._v("#")]),e._v(" get_original_quantity_resource "),a("Badge",{attrs:{text:"WithoutResourceAvailabilityChange",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_original_quantity_resource",sig:{params:[{name:"self"},{name:"resource",annotation:"str"},{name:"**kwargs"}],return:"int"}}}),e._v(" "),a("p",[e._v("Return the resource availability (int) for the given resource (either resource type or resource unit).")]),e._v(" "),a("h3",{attrs:{id:"get-quantity-resource-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-quantity-resource-3"}},[e._v("#")]),e._v(" get_quantity_resource "),a("Badge",{attrs:{text:"DeterministicResourceAvailabilityChanges",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_quantity_resource",sig:{params:[{name:"self"},{name:"resource",annotation:"str"},{name:"time",annotation:"int"},{name:"**kwargs"}],return:"int"}}}),e._v(" "),a("p",[e._v("Return the resource availability (int) for the given resource\n(either resource type or resource unit) at the given time.")]),e._v(" "),a("h3",{attrs:{id:"sample-quantity-resource-5"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-quantity-resource-5"}},[e._v("#")]),e._v(" sample_quantity_resource "),a("Badge",{attrs:{text:"UncertainResourceAvailabilityChanges",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"sample_quantity_resource",sig:{params:[{name:"self"},{name:"resource",annotation:"str"},{name:"time",annotation:"int"},{name:"**kwargs"}],return:"int"}}}),e._v(" "),a("p",[e._v("Sample an amount of resource availability (int) for the given resource\n(either resource type or resource unit) at the given time. This number should be the sum of the number of\nresource available at time t and the number of resource of this type consumed so far).")]),e._v(" "),a("h3",{attrs:{id:"get-original-quantity-resource-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-original-quantity-resource-2"}},[e._v("#")]),e._v(" _get_original_quantity_resource "),a("Badge",{attrs:{text:"WithoutResourceAvailabilityChange",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_get_original_quantity_resource",sig:{params:[{name:"self"},{name:"resource",annotation:"str"},{name:"**kwargs"}],return:"int"}}}),e._v(" "),a("p",[e._v("Return the resource availability (int) for the given resource (either resource type or resource unit).")]),e._v(" "),a("h3",{attrs:{id:"get-quantity-resource-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-quantity-resource-4"}},[e._v("#")]),e._v(" _get_quantity_resource "),a("Badge",{attrs:{text:"DeterministicResourceAvailabilityChanges",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_get_quantity_resource",sig:{params:[{name:"self"},{name:"resource",annotation:"str"},{name:"time",annotation:"int"},{name:"**kwargs"}],return:"int"}}}),e._v(" "),a("p",[e._v("Return the resource availability (int) for the given resource\n(either resource type or resource unit) at the given time.")]),e._v(" "),a("h3",{attrs:{id:"sample-quantity-resource-6"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-quantity-resource-6"}},[e._v("#")]),e._v(" _sample_quantity_resource "),a("Badge",{attrs:{text:"UncertainResourceAvailabilityChanges",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_sample_quantity_resource",sig:{params:[{name:"self"},{name:"resource",annotation:"str"},{name:"time",annotation:"int"},{name:"**kwargs"}],return:"int"}}}),e._v(" "),a("p",[e._v("Sample an amount of resource availability (int) for the given resource\n(either resource type or resource unit) at the given time. This number should be the sum of the number of\nresource available at time t and the number of resource of this type consumed so far).")])],1)}),[],!1,null,null,null);t.default=s.exports}}]);