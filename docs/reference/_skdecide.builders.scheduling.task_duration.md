# builders.scheduling.task_duration

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## SimulatedTaskDuration

A domain must inherit this class if the task duration requires sampling from a simulation.

### sample\_task\_duration <Badge text="SimulatedTaskDuration" type="tip"/>

<skdecide-signature name= "sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Sample, store and return task duration for the given task in the given mode.

### \_sample\_task\_duration <Badge text="SimulatedTaskDuration" type="tip"/>

<skdecide-signature name= "_sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return a task duration for the given task in the given mode.

## UncertainMultivariateTaskDuration

A domain must inherit this class if the task duration is uncertain and follows a know multivariate
distribution.

### get\_task\_duration\_distribution <Badge text="UncertainMultivariateTaskDuration" type="tip"/>

<skdecide-signature name= "get_task_duration_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}, {'name': 'multivariate_settings', 'default': 'None', 'annotation': 'Optional[Dict[str, int]]'}], 'return': 'Distribution'}"></skdecide-signature>

Return the multivariate Distribution of the duration of the given task in the given mode.
Multivariate seetings need to be provided. 

### sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return a task duration for the given task in the given mode,
sampled from the underlying multiivariate distribution.

### \_get\_task\_duration\_distribution <Badge text="UncertainMultivariateTaskDuration" type="tip"/>

<skdecide-signature name= "_get_task_duration_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}, {'name': 'multivariate_settings', 'default': 'None', 'annotation': 'Optional[Dict[str, int]]'}], 'return': 'Distribution'}"></skdecide-signature>

Return the multivariate Distribution of the duration of the given task in the given mode.
Multivariate seetings need to be provided. 

### \_sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "_sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return a task duration for the given task in the given mode,
sampled from the underlying multiivariate distribution.

## UncertainUnivariateTaskDuration

A domain must inherit this class if the task duration is uncertain and follows a know univariate distribution.

### get\_task\_duration\_distribution <Badge text="UncertainMultivariateTaskDuration" type="warn"/>

<skdecide-signature name= "get_task_duration_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}, {'name': 'multivariate_settings', 'default': 'None', 'annotation': 'Optional[Dict[str, int]]'}], 'return': 'Distribution'}"></skdecide-signature>

Return the multivariate Distribution of the duration of the given task in the given mode.
Multivariate seetings need to be provided. 

### sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return a task duration for the given task in the given mode,
sampled from the underlying multiivariate distribution.

### \_get\_task\_duration\_distribution <Badge text="UncertainMultivariateTaskDuration" type="warn"/>

<skdecide-signature name= "_get_task_duration_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}, {'name': 'multivariate_settings', 'default': 'None', 'annotation': 'Optional[Dict[str, int]]'}], 'return': 'Distribution'}"></skdecide-signature>

Return the univariate Distribution of the duration of the given task in the given mode.

### \_sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "_sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return a task duration for the given task in the given mode,
sampled from the underlying univariate distribution.

## UncertainBoundedTaskDuration

A domain must inherit this class if the task duration is known to be between a lower and upper bound
and follows a known distribution between these bounds.

### get\_task\_duration\_distribution <Badge text="UncertainMultivariateTaskDuration" type="warn"/>

<skdecide-signature name= "get_task_duration_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}, {'name': 'multivariate_settings', 'default': 'None', 'annotation': 'Optional[Dict[str, int]]'}], 'return': 'Distribution'}"></skdecide-signature>

Return the multivariate Distribution of the duration of the given task in the given mode.
Multivariate seetings need to be provided. 

### get\_task\_duration\_lower\_bound <Badge text="UncertainBoundedTaskDuration" type="tip"/>

<skdecide-signature name= "get_task_duration_lower_bound" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the lower bound for the task duration of the given task in the given mode.

### get\_task\_duration\_upper\_bound <Badge text="UncertainBoundedTaskDuration" type="tip"/>

<skdecide-signature name= "get_task_duration_upper_bound" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the upper bound for the task duration of the given task in the given mode.

### sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return a task duration for the given task in the given mode,
sampled from the underlying multiivariate distribution.

### \_get\_task\_duration\_distribution <Badge text="UncertainMultivariateTaskDuration" type="warn"/>

<skdecide-signature name= "_get_task_duration_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}, {'name': 'multivariate_settings', 'default': 'None', 'annotation': 'Optional[Dict[str, int]]'}], 'return': 'DiscreteDistribution'}"></skdecide-signature>

Return the Distribution of the duration of the given task in the given mode.
The distribution returns values beween the defined lower and upper bounds.

### \_get\_task\_duration\_lower\_bound <Badge text="UncertainBoundedTaskDuration" type="tip"/>

<skdecide-signature name= "_get_task_duration_lower_bound" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the lower bound for the task duration of the given task in the given mode.

### \_get\_task\_duration\_upper\_bound <Badge text="UncertainBoundedTaskDuration" type="tip"/>

<skdecide-signature name= "_get_task_duration_upper_bound" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the upper bound for the task duration of the given task in the given mode.

### \_sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "_sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return a task duration for the given task in the given mode,
sampled from the underlying univariate bounded distribution.

## UniformBoundedTaskDuration

A domain must inherit this class if the task duration is known to be between a lower and upper bound
and follows a uniform distribution between these bounds.

### get\_task\_duration\_distribution <Badge text="UncertainMultivariateTaskDuration" type="warn"/>

<skdecide-signature name= "get_task_duration_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}, {'name': 'multivariate_settings', 'default': 'None', 'annotation': 'Optional[Dict[str, int]]'}], 'return': 'Distribution'}"></skdecide-signature>

Return the multivariate Distribution of the duration of the given task in the given mode.
Multivariate seetings need to be provided. 

### get\_task\_duration\_lower\_bound <Badge text="UncertainBoundedTaskDuration" type="warn"/>

<skdecide-signature name= "get_task_duration_lower_bound" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the lower bound for the task duration of the given task in the given mode.

### get\_task\_duration\_upper\_bound <Badge text="UncertainBoundedTaskDuration" type="warn"/>

<skdecide-signature name= "get_task_duration_upper_bound" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the upper bound for the task duration of the given task in the given mode.

### sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return a task duration for the given task in the given mode,
sampled from the underlying multiivariate distribution.

### \_get\_task\_duration\_distribution <Badge text="UncertainMultivariateTaskDuration" type="warn"/>

<skdecide-signature name= "_get_task_duration_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}, {'name': 'multivariate_settings', 'default': 'None', 'annotation': 'Optional[Dict[str, int]]'}], 'return': 'DiscreteDistribution'}"></skdecide-signature>

Return the Distribution of the duration of the given task in the given mode.
The distribution is uniform between the defined lower and upper bounds.

### \_get\_task\_duration\_lower\_bound <Badge text="UncertainBoundedTaskDuration" type="warn"/>

<skdecide-signature name= "_get_task_duration_lower_bound" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the lower bound for the task duration of the given task in the given mode.

### \_get\_task\_duration\_upper\_bound <Badge text="UncertainBoundedTaskDuration" type="warn"/>

<skdecide-signature name= "_get_task_duration_upper_bound" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the upper bound for the task duration of the given task in the given mode.

### \_sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "_sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return a task duration for the given task in the given mode,
sampled from the underlying univariate uniform bounded distribution.

## EnumerableTaskDuration

A domain must inherit this class if the task duration for each task is enumerable.

### get\_task\_duration\_distribution <Badge text="UncertainMultivariateTaskDuration" type="warn"/>

<skdecide-signature name= "get_task_duration_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}, {'name': 'multivariate_settings', 'default': 'None', 'annotation': 'Optional[Dict[str, int]]'}], 'return': 'Distribution'}"></skdecide-signature>

Return the multivariate Distribution of the duration of the given task in the given mode.
Multivariate seetings need to be provided. 

### get\_task\_duration\_lower\_bound <Badge text="UncertainBoundedTaskDuration" type="warn"/>

<skdecide-signature name= "get_task_duration_lower_bound" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the lower bound for the task duration of the given task in the given mode.

### get\_task\_duration\_upper\_bound <Badge text="UncertainBoundedTaskDuration" type="warn"/>

<skdecide-signature name= "get_task_duration_upper_bound" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the upper bound for the task duration of the given task in the given mode.

### sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return a task duration for the given task in the given mode,
sampled from the underlying multiivariate distribution.

### \_get\_task\_duration\_distribution <Badge text="UncertainMultivariateTaskDuration" type="warn"/>

<skdecide-signature name= "_get_task_duration_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}, {'name': 'multivariate_settings', 'default': 'None', 'annotation': 'Optional[Dict[str, int]]'}], 'return': 'DiscreteDistribution'}"></skdecide-signature>

Return the Distribution of the duration of the given task in the given mode.
as an Enumerable.

### \_get\_task\_duration\_lower\_bound <Badge text="UncertainBoundedTaskDuration" type="warn"/>

<skdecide-signature name= "_get_task_duration_lower_bound" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the lower bound for the task duration of the given task in the given mode.

### \_get\_task\_duration\_upper\_bound <Badge text="UncertainBoundedTaskDuration" type="warn"/>

<skdecide-signature name= "_get_task_duration_upper_bound" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the upper bound for the task duration of the given task in the given mode.

### \_sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "_sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return a task duration for the given task in the given mode.

## DeterministicTaskDuration

A domain must inherit this class if the task durations are known and deterministic.

### get\_task\_duration <Badge text="DeterministicTaskDuration" type="tip"/>

<skdecide-signature name= "get_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the fixed deterministic task duration of the given task in the given mode.

### get\_task\_duration\_distribution <Badge text="UncertainMultivariateTaskDuration" type="warn"/>

<skdecide-signature name= "get_task_duration_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}, {'name': 'multivariate_settings', 'default': 'None', 'annotation': 'Optional[Dict[str, int]]'}], 'return': 'Distribution'}"></skdecide-signature>

Return the multivariate Distribution of the duration of the given task in the given mode.
Multivariate seetings need to be provided. 

### get\_task\_duration\_lower\_bound <Badge text="UncertainBoundedTaskDuration" type="warn"/>

<skdecide-signature name= "get_task_duration_lower_bound" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the lower bound for the task duration of the given task in the given mode.

### get\_task\_duration\_upper\_bound <Badge text="UncertainBoundedTaskDuration" type="warn"/>

<skdecide-signature name= "get_task_duration_upper_bound" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the upper bound for the task duration of the given task in the given mode.

### sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return a task duration for the given task in the given mode,
sampled from the underlying multiivariate distribution.

### \_get\_task\_duration <Badge text="DeterministicTaskDuration" type="tip"/>

<skdecide-signature name= "_get_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the fixed deterministic task duration of the given task in the given mode.

### \_get\_task\_duration\_distribution <Badge text="UncertainMultivariateTaskDuration" type="warn"/>

<skdecide-signature name= "_get_task_duration_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}, {'name': 'multivariate_settings', 'default': 'None', 'annotation': 'Optional[Dict[str, int]]'}]}"></skdecide-signature>

Return the Distribution of the duration of the given task in the given mode.
Because the duration is deterministic, the distribution always returns the same duration.

### \_get\_task\_duration\_lower\_bound <Badge text="UncertainBoundedTaskDuration" type="warn"/>

<skdecide-signature name= "_get_task_duration_lower_bound" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the lower bound for the task duration of the given task in the given mode.

### \_get\_task\_duration\_upper\_bound <Badge text="UncertainBoundedTaskDuration" type="warn"/>

<skdecide-signature name= "_get_task_duration_upper_bound" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return the upper bound for the task duration of the given task in the given mode.

### \_sample\_task\_duration <Badge text="SimulatedTaskDuration" type="warn"/>

<skdecide-signature name= "_sample_task_duration" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'mode', 'default': '1', 'annotation': 'Optional[int]'}, {'name': 'progress_from', 'default': '0.0', 'annotation': 'Optional[float]'}], 'return': 'int'}"></skdecide-signature>

Return a task duration for the given task in the given mode.

