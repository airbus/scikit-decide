%---------------------------------------------------------------------------%
% Multi-mode Resource-constrained Project Scheduling (MRCPSP)
%---------------------------------------------------------------------------%
% Authors: Ria Szeredi
%          Andreas Schutt
%
% Copyright: Data61, CSIRO.

% Adapted to multiskill multimode Project scheduling by AIR Airbus (2020)
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

    % Number of resources
int: n_res;
int: nb_res = n_res;


bool: one_ressource_per_task;
bool: exact_skills_need;
bool: add_calendar_constraint_unit;


    % Number of activities (tasks)
int: n_tasks;
int: nb_act = n_tasks;

    % Number of modes (optional activities)
int: n_opt;
int: nb_mod = n_opt;

    % Set of modes for each activity
array[Act] of set of Mod: modes;
array[Act] of set of Mod: mode = modes;

array[Mod] of int: dur;
array[Mod] of int: mdur = dur;

array[Res, Mod] of int: rreq;
array[Res, Mod] of int: mrreq = rreq;


array[Mod] of Act: mact = [sum(i in Act where m in mode[i])(i) | m in Mod];

%---------------------------------------------------------------------------%
% Parameters

set of int: Res = 1..nb_res; % Set of resources
array[Res] of int: rcap;     % Resource capacity
array[Res] of int: rtype;    % Resource type (1: renewable; 2: non-renewable)


int: nb_skill;
set of int: Skill = 1..nb_skill; % set of skills

int: nb_units;
set of int: Units = 1..nb_units; % set of units
array[Units, Skill] of int: skillunits;
array[Units, TIME_ARR] of int: ressource_unit_capacity_time;
array[Units, TIME_ARR0] of int: ressource_unit_capacity_time0 = array2d(Units, TIME_ARR0,
                                                                        ressource_unit_capacity_time);


array[Skill, Mod] of int: skillreq;
int: min_skillreq = min(k in Skill, m in Mod)(skillreq[k,m]);
int: max_skillreq = max(k in Skill, m in Mod)(skillreq[k,m])+10;
array[Skill, Act] of var min_skillreq..max_skillreq: array_skills_required = array2d(Skill, Act,
                                                                                     [let {var {skillreq[k, m] | m in mode[i]}: x} in x |
                                                                                      k in Skill, i in Act]);
array[Units, Act] of var bool: unit_used;


constraint(
if one_ressource_per_task then forall(i in Act)(sum(u in Units)(unit_used[u, i])<=1)
endif);


% Unit used available
constraint
    if add_calendar_constraint_unit then
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
    if exact_skills_need then
        forall(s in Skill, i in Act)(
            if array_skills_required[s, i]>0 then
            sum(w in Units)(skillunits[w, s]*unit_used[w, i])==array_skills_required[s, i]
            endif)
    endif;


var int: overskill=0;
%constraint overskill=sum(s in Skill, i in Act)(
%                        if array_skills_required[s,i]>0
%                           then sum(w in Units)(skillunits[w, s]*unit_used[w, i])-array_skills_required[s,i]
%                        else 0
%                        endif);

constraint
    forall(i in Act)(
        let { var mode[i]: mi } in (
            mrun[mi] = 1
        /\  forall(k in Skill)( array_skills_required[k,i] = skillreq[k, mi])
        )
    );


constraint forall(w in Units)(
     cumulative(start, adur,
                [unit_used[w, i] | i in Act],
                1)% [ressource_unit_capacity_time[w, t] | t in TIME_ARR])
);

%constraint forall(w in Units)(
%     my_cumulative(start, adur,
%                   [unit_used[w, i] | i in Act],
%                   [ressource_unit_capacity_time[w, t] | t in TIME_ARR],
%                   max_time)
%);


set of int: RRes = {k | k in Res where rtype[k] = 1};
set of int: NRes = {k | k in Res where rtype[k] = 2};
set of int: Act = 1..nb_act;    % Set of activities
set of int: Mod = 1..nb_mod;    % Set of modes
array[Act] of set of Act: succ; % Set of successors

int: min_mrreq = if nb_res>0 then min(k in Res, m in Mod)(mrreq[k,m]) else 0 endif;
int: max_mrreq = if nb_res>0 then max(k in Res, m in Mod)(mrreq[k,m]) else 0 endif;

%---------------------------------------------------------------------------%
% Variables
int: max_time;
int: UB = sum(i in Act)(max([mdur[m] | m in mode[i]]));
set of int: TIME_ARR=1..max_time+1;
set of int: TIME_ARR0=0..max_time;
array[Res, TIME_ARR] of int: ressource_capacity_time;
array[Res, TIME_ARR0] of int: ressource_capacity_time0 = array2d(Res, TIME_ARR0,
                                                                 ressource_capacity_time);


array[Mod] of var bool: mrun;
array[Act] of var TIME_ARR0: start;
output["\(UB)"];

array[Act] of var min(mdur)..max(mdur): adur = [let {var {mdur[m] | m in mode[i]}: x} in x | i in Act];
array[Res,Act] of var min_mrreq..max_mrreq: arreq = array2d(Res, Act,
                                                            [let {var {mrreq[k,m] | m in mode[i]}: x} in x |
                                                             k in Res, i in Act]);
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
        /\  forall(k in Res)( arreq[k,i] = mrreq[k, mi] )
        )
    );

    % Precedence constraints
    %
constraint forall(i in Act, j in succ[i])(start[i] + adur[i] <= start[j]);

    % Renewable resource constraints
    %
%constraint forall(k in RRes)(
%    my_cumulative(start, adur,
%                  [arreq[k,i] | i in Act],
%                  [ressource_capacity_time[k,t] | t in TIME_ARR])
%);

constraint forall(k in RRes)(
    cumulative(start, adur, [arreq[k,i] | i in Act], rcap[k])
);



    % Non-renewable resource constraints
    %
constraint forall(k in NRes)(sum(i in Act)(arreq[k,i]) <= rcap[k]);

    % Objective constraints
    %
constraint redundant_constraint(
    forall(i in Act where succ[i]={})(start[i] + adur[i] <= objective)
);
constraint objective = max(i in Act where succ[i]={})(start[i] + adur[i]);

%---------------------------------------------------------------------------%
% Non-overlapping constraints (NOC)

    % Options for NOC
    %
bool: opt_noc_half = false; % Use half-reification?

    % Check whether two activities are disjunct in all mode combinations
    %
function bool: are_act_disjunct(int: i, int: j) =
    forall(mi in mode[i], mj in mode[j])(are_mod_disjunct(mi, mj));

    % Check whether two modes are in disjunction
    %
function bool: are_mod_disjunct(int: mi, int: mj) =
    exists(k in RRes)(mrreq[k,mi] + mrreq[k,mj] > rcap[k]);

    % Check whether two activities can be disjunct in any mode combination
    %
function bool: can_act_be_disjunct(int: i, int: j) =
    exists(k in RRes)(
        can_act_be_disjunct_on_rres(i, j, k)
    );

    % Check whether two activities can be disjunct on a renewable resource
    % in any mode combination
    %
function bool: can_act_be_disjunct_on_rres(int: i, int: j, int: k) =
    exists(mi in mode[i], mj in mode[j])(
        mrreq[k,mi] + mrreq[k,mj] > rcap[k]
    );


    % Non-overlapping constraints for activities that are disjunct
    % in any mode combination
    %
predicate post_noc_disj(int: i, int: j) =
    if opt_noc_half then
        let {var bool: b} in (
            (    b  -> start[i] + adur[i] <= start[j])
        \/  (not(b) -> start[j] + adur[j] <= start[i])
        )
    else
        (start[i] + adur[i] <= start[j] \/ start[j] + adur[j] <= start[i])
    endif;

    % Non-overlapping constraints for different mode combinations
    % of potential disjunct activities
    %
predicate post_noc_mode(int: i, int: j) =
    forall(mi in mode[i], mj in mode[j] where are_mod_disjunct(mi, mj))(
        if opt_noc_half then
            let {var bool: b} in (
                ((mrun[mi] /\ mrun[mj]) -> (
                        (    b  -> start[i] + mdur[mi] <= start[j])
                    \/  (not(b) -> start[j] + mdur[mj] <= start[i])
                    )
                )
                % For fixing the Boolean variable in the case that not both modes are run
            /\  (not(mrun[mi] /\ mrun[mj]) -> b)
            )
        else
            (mrun[mi] /\ mrun[mj]) ->
                (start[i] + mdur[mi] <= start[j] \/ start[j] + mdur[mj] <= start[i])
        endif
    );

    % Non-overlapping constraints for different renewable
    % of potential disjunct activities
    %
predicate post_noc_rres(int: i, int: j) =
    forall(k in RRes where can_act_be_disjunct_on_rres(i, j, k))(
        if opt_noc_half then
            let {var bool: b} in (
                ((arreq[k,i] + arreq[k,j] > rcap[k]) -> (
                        (   (b) -> start[i] + adur[i] <= start[j])
                    \/  (not(b) -> start[j] + adur[j] <= start[i])
                    )
                )
                % For fixing the Boolean variable in the case that not both modes are run
            /\  (not(arreq[k,i] + arreq[k,j] <= rcap[k]) -> b)
            )
        else
            (arreq[k,i] + arreq[k,j] > rcap[k]) ->
                (start[i] + adur[i] <= start[j] \/ start[j] + adur[j] <= start[i])
        endif
    );

    % Non-overlapping constraints for different renewable
    % of potential disjunct activities via the cumulative constraint
    %
predicate post_noc_cumu(int: i, int: j) =
    forall(k in RRes where can_act_be_disjunct_on_rres(i, j, k))(
        cumulative([start[i], start[j]], [adur[i], adur[j]], [arreq[k,i], arreq[k,j]], rcap[k])
    );

    % Posting of the redundant NOCs
    %
constraint redundant_constraint(
    forall(i,j in Act where i < j /\ can_act_be_disjunct(i, j))(
        if are_act_disjunct(i, j) then
            % Activities are disjunct all mode combinations
            post_noc_disj(i, j)
        else
            % Activities might be disjunct
            post_noc_rres(i, j)
        endif
    )
);


%---------------------------------------------------------------------------%
% Solve item

solve
    :: durThenStartThenMode
    minimize objective;

ann: durThenStartThenMode = seq_search([
        int_search(start, smallest, indomain_min, complete),
        %int_search(adur, smallest, indomain_min, complete),
        %int_search(start, smallest, indomain_min, complete),
        bool_search(mrun, input_order, indomain_max, complete),
        bool_search(unit_used, input_order, indomain_min, complete)
    ]);

%---------------------------------------------------------------------------%
% Output

output [
    "start = \(start);\n",
    "mrun = \(mrun);\n",
    "objective = \(objective);\n",
    "array_skills_required = \(array_skills_required);\n"
];