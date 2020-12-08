int: nb_facilities;
int: nb_customers;
set of int: FACILITIES=1..nb_facilities;
set of int: CUSTOMERS=1..nb_customers;
array[FACILITIES, CUSTOMERS] of int: distance;
array[FACILITIES] of int: setup_cost_vector;
array[CUSTOMERS] of int: demand;
array[FACILITIES] of int: capacity;
array[CUSTOMERS] of var FACILITIES: facility_for_customer;
array[FACILITIES] of var int: capacity_used_by_facility;
array[FACILITIES] of var bool: used_facility;

constraint forall(f in FACILITIES)
              (sum(c in CUSTOMERS)(if facility_for_customer[c]==f then demand[c]
                                   else 0
                                   endif)==capacity_used_by_facility[f]);
constraint forall(f in FACILITIES)(capacity_used_by_facility[f]<=capacity[f]);
constraint forall(f in FACILITIES)(used_facility[f]==1 <-> capacity_used_by_facility[f]>0);

var int: objective;
var int: setup_cost;
var int: length_cost;
constraint setup_cost == sum(f in FACILITIES)(used_facility[f]*setup_cost_vector[f]);
constraint length_cost == sum(c in CUSTOMERS)(distance[facility_for_customer[c], c]);
constraint objective==setup_cost+length_cost;
output ["obj=\(objective)"];












