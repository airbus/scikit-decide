int: n_nodes;
int: n_edges;
int: nb_colors;
set of int: NODES = 1..n_nodes;
set of int: EDGES = 1..n_edges;
set of int: COLORS = 1..nb_colors;
array[EDGES, 1..2] of NODES: list_edges;
array[NODES] of set of NODES: graph=[{list_edges[e, 2]| e in EDGES where list_edges[e,1]=n} union 
                                     {list_edges[e, 1]| e in EDGES where list_edges[e,2]=n} | n in NODES];

include "alldifferent.mzn";
include "value_precede_chain.mzn";
%constraint value_precede_chain([c | c in COLORS],
%                                color_graph);
constraint forall(n in NODES)(forall(n_n in graph[n] where n_n>n)(color_graph[n]!=color_graph[n_n]));
array[NODES] of var COLORS: color_graph;
var int: obj=max(color_graph);
solve ::int_search(color_graph, smallest, indomain_min, complete) minimize(obj);
output ["obj=\(obj)"];


