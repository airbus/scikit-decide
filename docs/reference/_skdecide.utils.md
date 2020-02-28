# utils

This module contains utility functions.

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## rollout

<skdecide-signature name= "rollout" :sig="{'params': [{'name': 'domain', 'annotation': 'Domain'}, {'name': 'solver', 'default': 'None', 'annotation': 'Optional[Solver]'}, {'name': 'from_memory', 'default': 'None', 'annotation': 'Optional[D.T_memory[D.T_state]]'}, {'name': 'from_action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}, {'name': 'num_episodes', 'default': '1', 'annotation': 'int'}, {'name': 'max_steps', 'default': 'None', 'annotation': 'Optional[int]'}, {'name': 'render', 'default': 'True', 'annotation': 'bool'}, {'name': 'max_framerate', 'default': 'None', 'annotation': 'Optional[float]'}, {'name': 'verbose', 'default': 'True', 'annotation': 'bool'}, {'name': 'action_formatter', 'default': '<lambda function>', 'annotation': 'Optional[Callable[[D.T_event], str]]'}, {'name': 'outcome_formatter', 'default': '<lambda function>', 'annotation': 'Optional[Callable[[EnvironmentOutcome], str]]'}, {'name': 'save_result_directory', 'default': 'None', 'annotation': 'str'}], 'return': 'None'}"></skdecide-signature>

This method will run one or more episodes in a domain according to the policy of a solver.

#### Parameters
- **domain**: The domain in which the episode(s) will be run.
- **solver**: The solver whose policy will select actions to take (if None, a random policy is used).
- **from_memory**: The memory or state to consider as rollout starting point (if None, the domain is reset first).
- **from_action**: The last applied action when from_memory is used (if necessary for initial observation computation).
- **num_episodes**: The number of episodes to run.
- **max_steps**: The maximum number of steps for each episode (if None, no limit is set).
- **render**: Whether to render the episode(s) during rollout if the domain is renderable.
- **max_framerate**: The maximum number of steps/renders per second (if None, steps/renders are never slowed down).
- **verbose**: Whether to print information to the console during rollout.
- **action_formatter**: The function transforming actions in the string to print (if None, no print).
- **outcome_formatter**: The function transforming EnvironmentOutcome objects in the string to print (if None, no print).
- **save_result**: Directory in which state visited, actions applied and Transition Value are saved to json.

