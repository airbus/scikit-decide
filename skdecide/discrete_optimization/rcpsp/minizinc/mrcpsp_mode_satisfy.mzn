%---------------------------------------------------------------------------%
% Multi-mode Resource-constrained Project Scheduling (MRCPSP)
%---------------------------------------------------------------------------%
% Authors: Ria Szeredi
%          Andreas Schutt
%
% Copyright: Data61, CSIRO.
%---------------------------------------------------------------------------%
%
% This MRCPSP model was used in the publication of
%---------------------------------------------------------------------------%
% Includes

include "cumulative.mzn";
include "element.mzn";

%---------------------------------------------------------------------------%
% Mapping of parameters from the input data to the used parameter names in the
% publication

    % Number of resources
int: n_res;
int: nb_res = n_res;

    % Number of activities (tasks)
int: n_tasks;
int: nb_act = n_tasks;

    % Number of modes (optional activities)
int: n_opt;
int: nb_mod = n_opt;

    % Set of modes for each activity
array[Act] of set of Mod: modes;
array[Act] of set of Mod: mode = modes;

array[Res, Mod] of int: rreq;
array[Res, Mod] of int: mrreq = rreq;

array[Mod] of Act: mact = [sum(i in Act where m in mode[i])(i) | m in Mod];

%---------------------------------------------------------------------------%
% Parameters

set of int: Res = 1..nb_res; % Set of resources
array[Res] of int: rcap;     % Resource capacity
array[Res] of int: rtype;    % Resource type (1: renewable; 2: non-renewable)
set of int: RRes = {k | k in Res where rtype[k] = 1};
set of int: NRes = {k | k in Res where rtype[k] = 2};
set of int: Act = 1..nb_act;    % Set of activities
set of int: Mod = 1..nb_mod;    % Set of modes


int: min_mrreq = min(k in Res, m in Mod)(mrreq[k,m]);
int: max_mrreq = max(k in Res, m in Mod)(mrreq[k,m]);

%---------------------------------------------------------------------------%
% Variables
array[Mod] of var bool: mrun;
array[Res, Act] of var min_mrreq..max_mrreq: arreq = array2d(Res,Act,[let {var {mrreq[k,m] | m in mode[i]}: x} in x |
                                                                           k in Res, i in Act]);
%---------------------------------------------------------------------------%
% Constraints

    % Activities and mode constraints
    %
constraint
    forall(i in Act)(
        let { var mode[i]: mi } in (
            mrun[mi] = 1
        /\  sum(m in mode[i])(mrun[m]) = 1
        /\  forall(k in Res)( arreq[k,i] = mrreq[k, mi])
    ));


constraint forall(k in NRes)(sum(i in Act)(arreq[k, i]) <= rcap[k]);

constraint forall(k in RRes)(forall(i in Act)(arreq[k, i] <= rcap[k]));



solve satisfy;


output [
    "mrun = \(mrun);\n"
];
