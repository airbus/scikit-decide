# builders.discrete_optimization.generic_tools.do_mutation

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## LocalMoveDefault

Not clever local move
If you're lazy or don't have the choice,
don't do in place modification of the previous solution, so that you can retrieve it directly.
So the backward operator is then obvious.

### Constructor <Badge text="LocalMoveDefault" type="tip"/>

<skdecide-signature name= "LocalMoveDefault" :sig="{'params': [{'name': 'prev_solution', 'annotation': 'Solution'}, {'name': 'new_solution', 'annotation': 'Solution'}]}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

