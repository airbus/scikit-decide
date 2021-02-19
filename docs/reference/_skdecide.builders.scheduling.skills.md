# builders.scheduling.skills

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## WithResourceSkills

A domain must inherit this class if its resources (either resource types or resource units)
have different set of skills.

### find\_one\_ressource\_to\_do\_one\_task <Badge text="WithResourceSkills" type="tip"/>

<skdecide-signature name= "find_one_ressource_to_do_one_task" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'int'}], 'return': 'List[str]'}"></skdecide-signature>

For the common case when it is possible to do the task by one resource unit.
For general case, it might just return no possible ressource unit.

### get\_all\_resources\_skills <Badge text="WithResourceSkills" type="tip"/>

<skdecide-signature name= "get_all_resources_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, Dict[str, Any]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a resource type or resource unit
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {unit: {skill: (detail of skill)}} 

### get\_all\_tasks\_skills <Badge text="WithResourceSkills" type="tip"/>

<skdecide-signature name= "get_all_tasks_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, Dict[str, Any]]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a task
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {task: {skill: (detail of skill)}} 

### get\_skills\_names <Badge text="WithResourceSkills" type="tip"/>

<skdecide-signature name= "get_skills_names" :sig="{'params': [{'name': 'self'}], 'return': 'Set[str]'}"></skdecide-signature>

Return a list of all skill names as a list of str. Skill names are defined in the 2 dictionaries returned
by the get_all_resources_skills and get_all_tasks_skills functions.

### get\_skills\_of\_resource <Badge text="WithResourceSkills" type="tip"/>

<skdecide-signature name= "get_skills_of_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}], 'return': 'Dict[str, Any]'}"></skdecide-signature>

Return the skills of a given resource

### get\_skills\_of\_task <Badge text="WithResourceSkills" type="tip"/>

<skdecide-signature name= "get_skills_of_task" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'int'}], 'return': 'Dict[str, Any]'}"></skdecide-signature>

Return the skill requirements for a given task

### \_get\_all\_resources\_skills <Badge text="WithResourceSkills" type="tip"/>

<skdecide-signature name= "_get_all_resources_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, Dict[str, Any]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a resource type or resource unit
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {unit: {skill: (detail of skill)}} 

### \_get\_all\_tasks\_skills <Badge text="WithResourceSkills" type="tip"/>

<skdecide-signature name= "_get_all_tasks_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, Dict[str, Any]]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a task
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {task: {skill: (detail of skill)}} 

## WithoutResourceSkills

A domain must inherit this class if no resources skills have to be considered.

### find\_one\_ressource\_to\_do\_one\_task <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "find_one_ressource_to_do_one_task" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'int'}], 'return': 'List[str]'}"></skdecide-signature>

For the common case when it is possible to do the task by one resource unit.
For general case, it might just return no possible ressource unit.

### get\_all\_resources\_skills <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_all_resources_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, Dict[str, Any]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a resource type or resource unit
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {unit: {skill: (detail of skill)}} 

### get\_all\_tasks\_skills <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_all_tasks_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, Dict[str, Any]]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a task
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {task: {skill: (detail of skill)}} 

### get\_skills\_names <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_skills_names" :sig="{'params': [{'name': 'self'}], 'return': 'Set[str]'}"></skdecide-signature>

Return a list of all skill names as a list of str. Skill names are defined in the 2 dictionaries returned
by the get_all_resources_skills and get_all_tasks_skills functions.

### get\_skills\_of\_resource <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_skills_of_resource" :sig="{'params': [{'name': 'self'}, {'name': 'resource', 'annotation': 'str'}], 'return': 'Dict[str, Any]'}"></skdecide-signature>

Return the skills of a given resource

### get\_skills\_of\_task <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "get_skills_of_task" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'int'}], 'return': 'Dict[str, Any]'}"></skdecide-signature>

Return the skill requirements for a given task

### \_get\_all\_resources\_skills <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "_get_all_resources_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, Dict[str, Any]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a resource type or resource unit
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {unit: {skill: (detail of skill)}} 

### \_get\_all\_tasks\_skills <Badge text="WithResourceSkills" type="warn"/>

<skdecide-signature name= "_get_all_tasks_skills" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[str, Any]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the name of a task
and the second key is the name of a skill. The value defines the details of the skill.
 E.g. {task: {skill: (detail of skill)}} 

