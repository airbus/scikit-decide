predicate fzn_my_cumulative(array[int] of var int: s,
                            array[int] of var int: d,
                            array[int] of var int: r,
                            array[int] of int: b,
                            int: max_time) =
        let {
            set of int: Tasks =
                {i | i in index_set(s) where ub(r[i]) > 0 /\ ub(d[i]) > 0 }
        } in
            if 0==card(Tasks) then true %/*true*/ 0==card(index_set(s)) %\/ (not exists([c==0 | c in b]))
            else
                let {
                    int: early = min([ lb(s[i])            | i in Tasks ]),
                    int: late  = max([ ub(s[i]) + ub(d[i]) | i in Tasks ])
                } in (
                    %if late - early > 5000 then
                    %  fzn_cumulative_task(s, d, r, b)
                    %else
                    cumulative_time_calendar(s, d, r, b, max_time)
                    %endif
                )
            endif
            ;


predicate cumulative_time_calendar(array[int] of var int: s,
                                   array[int] of var int: d,
                                   array[int] of var int: r,
                                   array[int] of int: b,
                                   int: max_time) =
    let {
        set of int: Tasks =
            {i | i in index_set(s) where ub(r[i]) > 0 /\ ub(d[i]) > 0 },
        int: early = min([ lb(s[i]) | i in Tasks ]),
        int: late  = min(max_time, max([ ub(s[i]) + ub(d[i]) | i in Tasks ]))
    } in (
        forall(t in early..late)(
            b[t+1] >= sum( i in Tasks ) (
                bool2int(s[i] <= t /\ t < s[i] + d[i]) * r[i]
            )
        )
    );

% predicate fzn_cumulative_task(array[int] of var int: s,
%                               array[int] of var int: d,
%                               array[int] of var int: r,
%                               array[int] of var int: b) =
%     let {
%         set of int: Tasks =
%             {i | i in index_set(s) where ub(r[i]) > 0 /\ ub(d[i]) > 0 }
%     } in (
%         forall( j in Tasks ) (
%             b[t] >= r[j] + sum( i in Tasks where i != j ) (
%                 bool2int(s[i] <= s[j] /\ s[j] < s[i] + d[i] ) * r[i]
%             )
%         )
%     );
