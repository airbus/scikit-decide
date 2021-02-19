# builders.scheduling.resource_type

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## WithResourceTypes

A domain must inherit this class if some of its resources are resource types.

### get\_resource\_types\_names <Badge text="WithResourceTypes" type="tip"/>

<skdecide-signature name= "get_resource_types_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource types as a list.

### \_get\_resource\_types\_names <Badge text="WithResourceTypes" type="tip"/>

<skdecide-signature name= "_get_resource_types_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource types as a list.

## WithoutResourceTypes

A domain must inherit this class if it only uses resource types.

### get\_resource\_types\_names <Badge text="WithResourceTypes" type="warn"/>

<skdecide-signature name= "get_resource_types_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource types as a list.

### \_get\_resource\_types\_names <Badge text="WithResourceTypes" type="warn"/>

<skdecide-signature name= "_get_resource_types_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource types as a list.

## WithResourceUnits

A domain must inherit this class if some of its resources are resource units.

### get\_resource\_type\_for\_unit <Badge text="WithResourceUnits" type="tip"/>

<skdecide-signature name= "get_resource_type_for_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, str]'}"></skdecide-signature>

Return a dictionary where the key is a resource unit name and the value a resource type name.
An empty dictionary can be used if there are no resource unit matching a resource type.

### get\_resource\_units\_names <Badge text="WithResourceUnits" type="tip"/>

<skdecide-signature name= "get_resource_units_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource units as a list.

### \_get\_resource\_type\_for\_unit <Badge text="WithResourceUnits" type="tip"/>

<skdecide-signature name= "_get_resource_type_for_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, str]'}"></skdecide-signature>

Return a dictionary where the key is a resource unit name and the value a resource type name.
An empty dictionary can be used if there are no resource unit matching a resource type.

### \_get\_resource\_units\_names <Badge text="WithResourceUnits" type="tip"/>

<skdecide-signature name= "_get_resource_units_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource units as a list.

## SingleResourceUnit

A domain must inherit this class if there is no allocation to be done (i.e. there is a single resource).

### get\_resource\_type\_for\_unit <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "get_resource_type_for_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, str]'}"></skdecide-signature>

Return a dictionary where the key is a resource unit name and the value a resource type name.
An empty dictionary can be used if there are no resource unit matching a resource type.

### get\_resource\_units\_names <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "get_resource_units_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource units as a list.

### \_get\_resource\_type\_for\_unit <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "_get_resource_type_for_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, str]'}"></skdecide-signature>

Return a dictionary where the key is a resource unit name and the value a resource type name.
An empty dictionary can be used if there are no resource unit matching a resource type.

### \_get\_resource\_units\_names <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "_get_resource_units_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource units as a list.

## WithoutResourceUnit

A domain must inherit this class if it only uses resource types.

### get\_resource\_type\_for\_unit <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "get_resource_type_for_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, str]'}"></skdecide-signature>

Return a dictionary where the key is a resource unit name and the value a resource type name.
An empty dictionary can be used if there are no resource unit matching a resource type.

### get\_resource\_units\_names <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "get_resource_units_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource units as a list.

### \_get\_resource\_type\_for\_unit <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "_get_resource_type_for_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, str]'}"></skdecide-signature>

Return a dictionary where the key is a resource unit name and the value a resource type name.
An empty dictionary can be used if there are no resource unit matching a resource type.

### \_get\_resource\_units\_names <Badge text="WithResourceUnits" type="warn"/>

<skdecide-signature name= "_get_resource_units_names" :sig="{'params': [{'name': 'self'}], 'return': 'List[str]'}"></skdecide-signature>

Return the names (string) of all resource units as a list.

