include "globals.mzn";
include "circuit.mzn";
include "increasing.mzn";


int: nb_customers_without_depot;
int: nb_vehicle;
int: nb_customers_virtual = nb_customers_without_depot+nb_vehicle+1;
int: capacity_vehicle;
set of int: capacity_per_vehicle=0..capacity_vehicle;
set of int: customer_range=1..nb_customers_virtual;

array[1..nb_vehicle+1] of customer_range: virtual_nodes = [nb_customers_without_depot+i | i in 1..nb_vehicle+1];

array[customer_range, customer_range] of int: distance;
array[customer_range] of int: demand;
array[customer_range] of var customer_range: circuit_vrp;
array[1..nb_vehicle] of var customer_range: index_end_subcircuit;

constraint forall(v in 1..nb_vehicle)(circuit_vrp[index_end_subcircuit[v]]==virtual_nodes[v+1]);
array[1..nb_vehicle] of var capacity_per_vehicle: capa;
constraint forall(v in 2..nb_vehicle)(
    (sum(i in index_end_subcircuit[v-1]..index_end_subcircuit[v])(demand[circuit_vrp[i]]))==capa[v]);
constraint sum(i in 1..index_end_subcircuit[1])(demand[circuit_vrp[i]])==capa[1];
constraint circuit_vrp[1]==virtual_nodes[1];
constraint circuit_vrp[nb_customers_virtual]==virtual_nodes[nb_vehicle+1];
constraint all_different(circuit_vrp);
constraint increasing(index_end_subcircuit);
constraint decreasing([index_end_subcircuit[i+1]-index_end_subcircuit[i] | i in 1..nb_vehicle-1]);

var int: objective;
constraint objective==sum(i in 1..nb_customers_virtual-1)(distance[circuit_vrp[i], circuit_vrp[i+1]]);
output["obj=\(objective) \n"];
output["circuit =\(circuit_vrp) \n"];
%solve minimize(objective);