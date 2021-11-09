include "all_different.mzn";
include "disjunctive.mzn";
include "fzn_my_cumulative.mzn";
include "fzn_my_cumulative_reif.mzn";

/** @group globals.scheduling
  Requires that a set of tasks given by start times \a s, durations \a d, and
  resource requirements \a r, never require more than a global resource bound
  \a b at any one time.

  Assumptions:

  - forall \p i, \a d[\p i] >= 0 and \a r[\p i] >= 0
*/
predicate my_cumulative(array[int] of var int: s,
                        array[int] of var int: d,
                        array[int] of var int: r,
                        array[int] of int: b,
                        int: max_time) =
    fzn_my_cumulative(s, d, r, b, max_time);
