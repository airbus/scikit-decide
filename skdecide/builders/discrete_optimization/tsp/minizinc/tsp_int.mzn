%
% adaptation of https://github.com/hakank/hakank/blob/master/minizinc/tsp_circuit.mzn
% TSP with circuit predicate in Minizinc.
% 
% Model created by Hakan Kjellerstrand, hakank@gamil.com
% with help from Krzysztof Kuchcinski (of the JaCoP team)
%                 
% There are two optimal solutions (with distance=6859):
%
%   x: [14, 4, 2, 8, 11, 15, 6, 1, 10, 16, 9, 7, 12, 13, 5, 3]
%   6859
%   ----------
%   x: [8, 3, 16, 2, 15, 7, 12, 4, 11, 9, 5, 13, 14, 1, 6, 10]
%   6859
%
%
% Also, see my MiniZinc page: http://www.hakank.org/minizinc/
%

include "globals.mzn";
% include "jacop.mzn";

predicate circuit_me(array[int] of var int: x) =
  let { 
        int: lbx = 1,
        int: ubx = length(x),
        array[lbx..ubx] of var lbx..ubx: z
  }
  in
   all_different(x) :: domain /\
   all_different(z) :: domain /\

   % put the orbit of x[1] in in z[1..n]
   % 
   z[lbx] = x[lbx] /\ 
   forall(i in lbx+1..ubx) (
      z[i] = x[z[i-1]]
   )
   /\ % may not be 1 for i < n
   forall(i in lbx..ubx-1) (
      z[i] != lbx
   )
   /\  % when i = n it must be 1
   z[ubx] = lbx
;

%
% Data
%
int: n; % number of nodes
set of int: NODES=1..n;
NODES: start;
NODES: end;
array[1..n] of var NODES: x; % the circuit
array[NODES, NODES] of int: distances;  % distance matrix
int: min_val = min([distances[i,j] | i,j in 1..n where distances[i,j] >= 0]);
int: max_val = max([distances[i,j] | i,j in 1..n]);
array[NODES] of var min_val..max_val: d;
var int: distance = sum(d);
%solve minimize distance;
solve :: int_search(d, max_regret, indomain_split, complete) minimize distance;
constraint if end!=start then x[end]=start endif;
constraint
    all_different(x) /\ % this speeds up for some solvers
    circuit(x) % use this for solvers with built-in circuit predicate, e.g. JaCoP
               % or - lately - the built-in version in MiniZinc distribution
    % circuit_me(x) % hakank: use this version for solvers without the built-in circuit predicate
    /\
    forall(i in 1..n) (
      distances[i, x[i]] = d[i]
    )
;
