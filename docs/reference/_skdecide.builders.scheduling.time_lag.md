# builders.scheduling.time_lag

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## TimeLag

Defines a time lag with both a minimum time lag and maximum time lag.

### Constructor <Badge text="TimeLag" type="tip"/>

<skdecide-signature name= "TimeLag" :sig="{'params': [{'name': 'minimum_time_lag'}, {'name': 'maximum_time_lags'}]}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

## MinimumOnlyTimeLag

Defines a minimum time lag.

### Constructor <Badge text="MinimumOnlyTimeLag" type="tip"/>

<skdecide-signature name= "MinimumOnlyTimeLag" :sig="{'params': [{'name': 'minimum_time_lag'}]}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

## MaximumOnlyTimeLag

Defines a maximum time lag.

### Constructor <Badge text="MaximumOnlyTimeLag" type="tip"/>

<skdecide-signature name= "MaximumOnlyTimeLag" :sig="{'params': [{'name': 'maximum_time_lags'}]}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

## WithTimeLag

A domain must inherit this class if there are minimum and maximum time lags between some of its tasks.

### get\_time\_lags <Badge text="WithTimeLag" type="tip"/>

<skdecide-signature name= "get_time_lags" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, TimeLag]]'}"></skdecide-signature>

Return nested dictionaries where the first key is the id of a task (int)
and the second key is the id of another task (int).
The value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end
of the first task to the start of the second task.

e.g.
    {
        12:{
            15: TimeLag(5, 10),
            16: TimeLag(5, 20),
            17: MinimumOnlyTimeLag(5),
            18: MaximumOnlyTimeLag(15),
        }
    }

#### Returns
A dictionary of TimeLag objects.

### \_get\_time\_lags <Badge text="WithTimeLag" type="tip"/>

<skdecide-signature name= "_get_time_lags" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, TimeLag]]'}"></skdecide-signature>

Return nested dictionaries where the first key is the id of a task (int)
and the second key is the id of another task (int).
The value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end
of the first task to the start of the second task.

e.g.
    {
        12:{
            15: TimeLag(5, 10),
            16: TimeLag(5, 20),
            17: MinimumOnlyTimeLag(5),
            18: MaximumOnlyTimeLag(15),
        }
    }

#### Returns
A dictionary of TimeLag objects.

## WithoutTimeLag

A domain must inherit this class if there is no required time lag between its tasks.

### get\_time\_lags <Badge text="WithTimeLag" type="warn"/>

<skdecide-signature name= "get_time_lags" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, TimeLag]]'}"></skdecide-signature>

Return nested dictionaries where the first key is the id of a task (int)
and the second key is the id of another task (int).
The value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end
of the first task to the start of the second task.

e.g.
    {
        12:{
            15: TimeLag(5, 10),
            16: TimeLag(5, 20),
            17: MinimumOnlyTimeLag(5),
            18: MaximumOnlyTimeLag(15),
        }
    }

#### Returns
A dictionary of TimeLag objects.

### \_get\_time\_lags <Badge text="WithTimeLag" type="warn"/>

<skdecide-signature name= "_get_time_lags" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, TimeLag]]'}"></skdecide-signature>

Return nested dictionaries where the first key is the id of a task (int)
and the second key is the id of another task (int).
The value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end
of the first task to the start of the second task.

