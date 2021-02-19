# builders.scheduling.time_windows

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## TimeWindow

Defines a time window with earliest start, latest start, earliest end and latest end only.

### Constructor <Badge text="TimeWindow" type="tip"/>

<skdecide-signature name= "TimeWindow" :sig="{'params': [{'name': 'earliest_start', 'annotation': 'int'}, {'name': 'latest_start', 'annotation': 'int'}, {'name': 'earliest_end', 'annotation': 'int'}, {'name': 'latest_end', 'annotation': 'int'}, {'name': 'max_horizon', 'annotation': 'int'}], 'return': 'None'}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

## ClassicTimeWindow

Defines a time window with earliest start and latest end only.

### Constructor <Badge text="ClassicTimeWindow" type="tip"/>

<skdecide-signature name= "ClassicTimeWindow" :sig="{'params': [{'name': 'earliest_start', 'annotation': 'int'}, {'name': 'latest_end', 'annotation': 'int'}, {'name': 'max_horizon', 'annotation': 'int'}], 'return': 'None'}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

## StartFromOnlyTimeWindow

Defines a time window with an earliest start only.

### Constructor <Badge text="StartFromOnlyTimeWindow" type="tip"/>

<skdecide-signature name= "StartFromOnlyTimeWindow" :sig="{'params': [{'name': 'earliest_start', 'annotation': 'int'}, {'name': 'max_horizon', 'annotation': 'int'}], 'return': 'None'}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

## StartBeforeOnlyTimeWindow

Defines a time window with an latest start only.

### Constructor <Badge text="StartBeforeOnlyTimeWindow" type="tip"/>

<skdecide-signature name= "StartBeforeOnlyTimeWindow" :sig="{'params': [{'name': 'latest_start', 'annotation': 'int'}, {'name': 'max_horizon', 'annotation': 'int'}], 'return': 'None'}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

## EndFromOnlyTimeWindow

Defines a time window with an earliest end only.

### Constructor <Badge text="EndFromOnlyTimeWindow" type="tip"/>

<skdecide-signature name= "EndFromOnlyTimeWindow" :sig="{'params': [{'name': 'earliest_end', 'annotation': 'int'}, {'name': 'max_horizon', 'annotation': 'int'}], 'return': 'None'}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

## EndBeforeOnlyTimeWindow

Defines a time window with a latest end only.

### Constructor <Badge text="EndBeforeOnlyTimeWindow" type="tip"/>

<skdecide-signature name= "EndBeforeOnlyTimeWindow" :sig="{'params': [{'name': 'latest_end', 'annotation': 'int'}, {'name': 'max_horizon', 'annotation': 'int'}], 'return': 'None'}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

## StartTimeWindow

Defines a time window with an earliest start and a latest start only.

### Constructor <Badge text="StartTimeWindow" type="tip"/>

<skdecide-signature name= "StartTimeWindow" :sig="{'params': [{'name': 'earliest_start', 'annotation': 'int'}, {'name': 'latest_start', 'annotation': 'int'}, {'name': 'max_horizon', 'annotation': 'int'}], 'return': 'None'}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

## EndTimeWindow

Defines a time window with an earliest end and a latest end only.

### Constructor <Badge text="EndTimeWindow" type="tip"/>

<skdecide-signature name= "EndTimeWindow" :sig="{'params': [{'name': 'earliest_end', 'annotation': 'int'}, {'name': 'latest_end', 'annotation': 'int'}, {'name': 'max_horizon', 'annotation': 'int'}], 'return': 'None'}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

## EmptyTimeWindow

Defines an empty time window.

### Constructor <Badge text="EmptyTimeWindow" type="tip"/>

<skdecide-signature name= "EmptyTimeWindow" :sig="{'params': [{'name': 'max_horizon', 'annotation': 'int'}], 'return': 'None'}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

## WithTimeWindow

A domain must inherit this class if some tasks have time windows defined.

### get\_time\_window <Badge text="WithTimeWindow" type="tip"/>

<skdecide-signature name= "get_time_window" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, TimeWindow]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value is a TimeWindow object.
Note that the max time horizon needs to be provided to the TimeWindow constructors
e.g.
    {
        1: TimeWindow(10, 15, 20, 30, self.get_max_horizon())
        2: EmptyTimeWindow(self.get_max_horizon())
        3: EndTimeWindow(20, 25, self.get_max_horizon())
        4: EndBeforeOnlyTimeWindow(40, self.get_max_horizon())
    }

#### Returns
A dictionary of TimeWindow objects.

### \_get\_time\_window <Badge text="WithTimeWindow" type="tip"/>

<skdecide-signature name= "_get_time_window" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, TimeWindow]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value is a TimeWindow object.
Note that the max time horizon needs to be provided to the TimeWindow constructors
e.g.
    {
        1: TimeWindow(10, 15, 20, 30, self.get_max_horizon())
        2: EmptyTimeWindow(self.get_max_horizon())
        3: EndTimeWindow(20, 25, self.get_max_horizon())
        4: EndBeforeOnlyTimeWindow(40, self.get_max_horizon())
    }

#### Returns
A dictionary of TimeWindow objects.

## WithoutTimeWindow

A domain must inherit this class if none of the tasks have restrictions on start times or end times.

### get\_time\_window <Badge text="WithTimeWindow" type="warn"/>

<skdecide-signature name= "get_time_window" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, TimeWindow]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value is a TimeWindow object.
Note that the max time horizon needs to be provided to the TimeWindow constructors
e.g.
    {
        1: TimeWindow(10, 15, 20, 30, self.get_max_horizon())
        2: EmptyTimeWindow(self.get_max_horizon())
        3: EndTimeWindow(20, 25, self.get_max_horizon())
        4: EndBeforeOnlyTimeWindow(40, self.get_max_horizon())
    }

#### Returns
A dictionary of TimeWindow objects.

### \_get\_time\_window <Badge text="WithTimeWindow" type="warn"/>

<skdecide-signature name= "_get_time_window" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, TimeWindow]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value is a dictionary of EmptyTimeWindow object.

#### Returns
A dictionary of TimeWindow objects.

