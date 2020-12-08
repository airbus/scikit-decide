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
include "my_cumulative.mzn";

%---------------------------------------------------------------------------%
% Mapping of parameters from the input data to the used parameter names in the
% publication


    % Number of activities (tasks)
int: n_tasks;
int: nb_act = n_tasks;

bool: one_ressource_per_task;
bool: ressource_always_here = true;
    % Number of modes (optional activities)
int: n_opt;
int: nb_mod = n_opt;

    % Set of modes for each activity
array[Act] of set of Mod: modes;
array[Act] of set of Mod: mode = modes;

array[Mod] of int: dur;
array[Mod] of int: mdur = dur;

array[Mod] of Act: mact = [sum(i in Act where m in mode[i])(i) | m in Mod];

%---------------------------------------------------------------------------%
% Parameters


int: nb_skill;
set of int: Skill = 1..nb_skill; % set of skills

int: nb_units;
set of int: Units = 1..nb_units; % set of units
array[Units, Skill] of int: skillunits;
array[Units, TIME_ARR] of int: ressource_unit_capacity_time;
array[Units, TIME_ARR0] of int: ressource_unit_capacity_time0=array2d(Units, TIME_ARR0, ressource_unit_capacity_time);



array[Skill, Mod] of int: skillreq;
int: min_skillreq = min(k in Skill, m in Mod)(skillreq[k,m]);
int: max_skillreq = max(k in Skill, m in Mod)(skillreq[k,m])+10;
array[Skill, Act] of var min_skillreq..max_skillreq: array_skills_required = array2d(Skill, Act,
                                                                                     [let {var {skillreq[k, m] | m in mode[i]}: x} in x |
                                                                                      k in Skill, i in Act]);
array[Units, Act] of var bool: unit_used;

constraint(
if one_ressource_per_task then forall(i in Act)((sum(u in Units)(unit_used[u, i]))<=1)
endif);

%constraint(
%if one_ressource_per_task then forall(i in Act)((sum(u in Units)(unit_used[u, i]))==1)
%endif);

constraint(
if one_ressource_per_task then
forall(skill in Skill, m in Mod, u in Units)(
if skillunits[u, skill]<skillreq[skill, m] then unit_used[u, mact[m]]=0
endif
)
endif
);

% Unit used available
constraint 
if not ressource_always_here then
forall(i in Act)(
     forall(w in Units)(
     let{var TIME_ARR0: st=start[i],
         var TIME_ARR0: end=start[i]+adur[i]} in (
            unit_used[w, i] <= min(t in st..end-1)(ressource_unit_capacity_time0[w, t])
         )
       )
      )
endif;

constraint forall(s in Skill, i in Act)(
    sum(w in Units)(skillunits[w, s]*unit_used[w, i])>=array_skills_required[s, i]); % skills used overachieve
                                                                                     % the need of the skill.
constraint
    forall(i in Act)(
        let { var mode[i]: mi } in (
            mrun[mi] = 1
        /\  forall(k in Skill)( array_skills_required[k,i] = skillreq[k, mi])
        )
    );


constraint forall(w in Units)(
     cumulative(start, [adur[i] | i in Act],
                [unit_used[w, i] | i in Act],
                1)
);

%constraint forall(w in Units)(
%     my_cumulative(start, adur,
%                   [unit_used[w, i] | i in Act],
%                   [ressource_unit_capacity_time[w, t] | t in TIME_ARR])
%);

set of int: Act = 1..nb_act;    % Set of activities
set of int: Mod = 1..nb_mod;    % Set of modes
array[Act] of set of Act: succ; % Set of successors


%---------------------------------------------------------------------------%
% Variables
int: max_time;
%int: UB = sum(i in Act)(max([mdur[m] | m in mode[i]]));
int: UB = max_time;

set of int: TIME_ARR=1..max_time+1;
set of int: TIME_ARR0=0..max_time;


array[Mod] of var bool: mrun;
array[Act] of var 0..UB: start;
output["\(UB)"];

array[Act] of var min(mdur)..max(mdur): adur = [let {var {mdur[m] | m in mode[i]}: x} in x | i in Act];
var 0..UB: objective;

%---------------------------------------------------------------------------%
% Constraints

    % Activities and mode constraints
    %
constraint
    forall(i in Act)(
        let { var mode[i]: mi } in (
            mrun[mi] = 1
        /\  sum(m in mode[i])(mrun[m]) = 1
        /\  adur[i] = mdur[mi]
        )
    );

    % Precedence constraints
    %
constraint forall(i in Act, j in succ[i])(start[i] + adur[i] <= start[j]);

    % Objective constraints
    %
constraint redundant_constraint(
    forall(i in Act where succ[i]={})(start[i] + adur[i] <= objective)
);
constraint objective = max(i in Act where succ[i]={})(start[i] + adur[i]);


%---------------------------------------------------------------------------%
% Solve item

solve
    :: durThenStartThenMode
    minimize objective;

ann: durThenStartThenMode = seq_search([
        %bool_search(unit_used, input_order, indomain_max, complete),
        int_search(start, smallest, indomain_min, complete),
        int_search(adur, smallest, indomain_min, complete),
        bool_search(mrun, input_order, indomain_max, complete),
    ]);

%---------------------------------------------------------------------------%
% Output

output [
    "start = \(start);\n",
    "mrun = \(mrun);\n",
    "objective = \(objective);\n",
    "array_skills_required = \(array_skills_required);\n"
];