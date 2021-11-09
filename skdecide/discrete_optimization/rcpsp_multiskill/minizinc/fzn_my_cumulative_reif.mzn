predicate fzn_my_cumulative_reif(array[int] of var int: s,
                                 array[int] of var int: d,
                                 array[int] of var int: r,
                                 array[int] of int: b,
                                 var bool: bb) =
        let {
            set of int: Tasks =
                {i | i in index_set(s) where ub(r[i]) > 0 /\ ub(d[i]) > 0 }
        } in
            if 0==card(Tasks) then true  % */ 0==card(index_set(s)) %\/ (not exists([c==0 | c in b]))
            else
                let {
                    int: early = min([ lb(s[i])            | i in Tasks ]),
                    int: late  = max([ ub(s[i]) + ub(d[i]) | i in Tasks ])
                } in (
                    %if late - early > 5000 then
                    %  fzn_cumulative_task(s, d, r, b)
                    %else
                    cumulative_time_calendar_reif(s, d, r, b, bb)
                    %endif
                )
            endif
            ;


predicate cumulative_time_calendar_reif(array[int] of var int: s,
                                        array[int] of var int: d,
                                        array[int] of var int: r,
                                        array[int] of int: b,
                                        var bool: bb) =
    let {
        set of int: Tasks =
            {i | i in index_set(s) where ub(r[i]) > 0 /\ ub(d[i]) > 0 },
        int: early = min([ lb(s[i]) | i in Tasks ]),
        int: late  = max([ ub(s[i]) + ub(d[i]) | i in Tasks ])
    } in (
        bb <-> forall( t in early..late ) (
            b[t] >= sum( i in Tasks ) (
                bool2int(s[i] <= t /\ t < s[i] + d[i]) * r[i]
            )
        )
    );
