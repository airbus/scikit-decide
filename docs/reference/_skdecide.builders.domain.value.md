# builders.domain.value

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## Rewards

A domain must inherit this class if it sends rewards (positive and/or negative).

### check\_value <Badge text="Rewards" type="tip"/>

<skdecide-signature name= "check_value" :sig="{'params': [{'name': 'self'}, {'name': 'value', 'annotation': 'TransitionValue[D.T_value]'}], 'return': 'bool'}"></skdecide-signature>

Check that a transition value is compliant with its reward specification.

::: tip
This function returns always True by default because any kind of reward should be accepted at this level.
:::

#### Parameters
- **value**: The transition value to check.

#### Returns
True if the transition value is compliant (False otherwise).

### \_check\_value <Badge text="Rewards" type="tip"/>

<skdecide-signature name= "_check_value" :sig="{'params': [{'name': 'self'}, {'name': 'value', 'annotation': 'TransitionValue[D.T_value]'}], 'return': 'bool'}"></skdecide-signature>

Check that a transition value is compliant with its reward specification.

::: tip
This function returns always True by default because any kind of reward should be accepted at this level.
:::

#### Parameters
- **value**: The transition value to check.

#### Returns
True if the transition value is compliant (False otherwise).

## PositiveCosts

A domain must inherit this class if it sends only positive costs (i.e. negative rewards).

Having only positive costs is a required assumption for certain solvers to work, such as classical planners.

### check\_value <Badge text="Rewards" type="warn"/>

<skdecide-signature name= "check_value" :sig="{'params': [{'name': 'self'}, {'name': 'value', 'annotation': 'TransitionValue[D.T_value]'}], 'return': 'bool'}"></skdecide-signature>

Check that a transition value is compliant with its reward specification.

::: tip
This function returns always True by default because any kind of reward should be accepted at this level.
:::

#### Parameters
- **value**: The transition value to check.

#### Returns
True if the transition value is compliant (False otherwise).

### \_check\_value <Badge text="Rewards" type="warn"/>

<skdecide-signature name= "_check_value" :sig="{'params': [{'name': 'self'}, {'name': 'value', 'annotation': 'TransitionValue[D.T_value]'}], 'return': 'bool'}"></skdecide-signature>

Check that a transition value is compliant with its cost specification (must be positive).

::: tip
This function calls `PositiveCost._is_positive()` to determine if a value is positive (can be overridden for
advanced value types).
:::

#### Parameters
- **value**: The transition value to check.

#### Returns
True if the transition value is compliant (False otherwise).

### \_is\_positive <Badge text="PositiveCosts" type="tip"/>

<skdecide-signature name= "_is_positive" :sig="{'params': [{'name': 'self'}, {'name': 'cost', 'annotation': 'D.T_value'}], 'return': 'bool'}"></skdecide-signature>

Determine if a value is positive (can be overridden for advanced value types).

#### Parameters
- **cost**: The cost to evaluate.

#### Returns
True if the cost is positive (False otherwise).

