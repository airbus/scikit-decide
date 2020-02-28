# hub.solver.cgp.cgp

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## flatten

<skdecide-signature name= "flatten" :sig="{'params': [{'name': 'c'}]}"></skdecide-signature>

Generator flattening the structure

>>> list(flatten([2, [2, "test", [4,5, [7], [2, [6, 2, 6, [6], 4]], 6]]]))
[2, 2, "test", 4, 5, 7, 2, 6, 2, 6, 6, 4, 6]

## norm\_and\_flatten

<skdecide-signature name= "norm_and_flatten" :sig="{'params': [{'name': 'vals'}, {'name': 'types'}]}"></skdecide-signature>

Flatten and normalise according to AIGYM type (BOX, DISCRETE, TUPLE)
:param vals: a np array structure
:param types: the gym type corresponding to the vals arrays
:return: a flatten array with normalised vals

## denorm

<skdecide-signature name= "denorm" :sig="{'params': [{'name': 'vals'}, {'name': 'types'}]}"></skdecide-signature>

Denormalize values according to AIGYM types (BOX, DISCRETE, TUPLE)
:param vals: an array of [-1,1] normalised values
:param types: the gym types corresponding to vals
:return: the same vals array with denormalised values

