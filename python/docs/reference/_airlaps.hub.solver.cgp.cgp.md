# hub.solver.cgp.cgp

[[toc]]

## flatten

<airlaps-signature name= "flatten" :sig="{'params': [{'name': 'c'}]}"></airlaps-signature>

Generator flattening the structure

\>\>\> list(flatten([2, [2, "test", (4, 5, [7], [2, [6, 2, 6, [6], 4]], 6)]]))
[2, 2, "test", 4, 5, 7, 2, 6, 2, 6, 6, 4, 6]

