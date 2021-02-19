# builders.scheduling.resource_availability

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## UncertainResourceAvailabilityChanges

A domain must inherit this class if the availability of its resource vary in an uncertain way over time.

### check\_unique\_resource\_names <Badge text="UncertainResourceAvailabilityChanges" type="tip"/>

<skdecide-signature name= "check_unique_resource_names" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Return True if there are no duplicates in resource names across both resource types
and resource units name lists.

### sample\_quantity\_resource <Badge text="UncertainResourceAvailabilityChanges" type="tip"/>

<skdecide-signature name= "sample_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Sample an amount of resource availability (int) for the given resource
(either resource type or resource unit) at the given time. This number should be the sum of the number of
resource available at time t and the number of resource of this type consumed so far).

### \_sample\_quantity\_resource <Badge text="UncertainResourceAvailabilityChanges" type="tip"/>

<skdecide-signature name= "_sample_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Sample an amount of resource availability (int) for the given resource
(either resource type or resource unit) at the given time. This number should be the sum of the number of
resource available at time t and the number of resource of this type consumed so far).

## DeterministicResourceAvailabilityChanges

A domain must inherit this class if the availability of its resource vary in a deterministic way over time.

### check\_unique\_resource\_names <Badge text="UncertainResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "check_unique_resource_names" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Return True if there are no duplicates in resource names across both resource types
and resource units name lists.

### get\_quantity\_resource <Badge text="DeterministicResourceAvailabilityChanges" type="tip"/>

<skdecide-signature name= "get_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Return the resource availability (int) for the given resource
(either resource type or resource unit) at the given time.

### sample\_quantity\_resource <Badge text="UncertainResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "sample_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Sample an amount of resource availability (int) for the given resource
(either resource type or resource unit) at the given time. This number should be the sum of the number of
resource available at time t and the number of resource of this type consumed so far).

### \_get\_quantity\_resource <Badge text="DeterministicResourceAvailabilityChanges" type="tip"/>

<skdecide-signature name= "_get_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Return the resource availability (int) for the given resource
(either resource type or resource unit) at the given time.

### \_sample\_quantity\_resource <Badge text="UncertainResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "_sample_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Sample an amount of resource availability (int) for the given resource
(either resource type or resource unit) at the given time. This number should be the sum of the number of
resource available at time t and the number of resource of this type consumed so far).

## WithoutResourceAvailabilityChange

A domain must inherit this class if the availability of its resource does not vary over time.

### check\_unique\_resource\_names <Badge text="UncertainResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "check_unique_resource_names" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Return True if there are no duplicates in resource names across both resource types
and resource units name lists.

### get\_original\_quantity\_resource <Badge text="WithoutResourceAvailabilityChange" type="tip"/>

<skdecide-signature name= "get_original_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Return the resource availability (int) for the given resource (either resource type or resource unit).

### get\_quantity\_resource <Badge text="DeterministicResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "get_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Return the resource availability (int) for the given resource
(either resource type or resource unit) at the given time.

### sample\_quantity\_resource <Badge text="UncertainResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "sample_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Sample an amount of resource availability (int) for the given resource
(either resource type or resource unit) at the given time. This number should be the sum of the number of
resource available at time t and the number of resource of this type consumed so far).

### \_get\_original\_quantity\_resource <Badge text="WithoutResourceAvailabilityChange" type="tip"/>

<skdecide-signature name= "_get_original_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Return the resource availability (int) for the given resource (either resource type or resource unit).

### \_get\_quantity\_resource <Badge text="DeterministicResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "_get_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Return the resource availability (int) for the given resource
(either resource type or resource unit) at the given time.

### \_sample\_quantity\_resource <Badge text="UncertainResourceAvailabilityChanges" type="warn"/>

<skdecide-signature name= "_sample_quantity_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}, {'name': 'kwargs'}], 'return': 'int'}"></skdecide-signature>

Sample an amount of resource availability (int) for the given resource
(either resource type or resource unit) at the given time. This number should be the sum of the number of
resource available at time t and the number of resource of this type consumed so far).

