# builders.discrete_optimization.generic_tools.ea.nsga

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## Nsga

NSGA

Args:
    problem:
        the problem to solve
    encoding:
        name (str) of an encoding registered in the register solution of Problem
        or a dictionary of the form {'type': TypeAttribute, 'n': int} where type refers to a TypeAttribute and n
         to the dimension of the problem in this encoding (e.g. length of the vector)
        by default, the first encoding in the problem register_solution will be used.

### Constructor <Badge text="Nsga" type="tip"/>

<skdecide-signature name= "Nsga" :sig="{'params': [{'name': 'problem', 'annotation': 'Problem'}, {'name': 'mutation', 'default': 'None', 'annotation': 'Union[Mutation, DeapMutation]'}, {'name': 'crossover', 'default': 'None', 'annotation': 'DeapCrossover'}, {'name': 'selection', 'default': 'None', 'annotation': 'DeapSelection'}, {'name': 'encoding', 'default': 'None', 'annotation': 'Optional[Union[str, Dict[str, Any]]]'}, {'name': 'objectives', 'default': 'None', 'annotation': 'Optional[Union[str, List[str]]]'}, {'name': 'objective_weights', 'default': 'None', 'annotation': 'Optional[List[float]]'}, {'name': 'pop_size', 'default': 'None', 'annotation': 'int'}, {'name': 'max_evals', 'default': 'None', 'annotation': 'int'}, {'name': 'mut_rate', 'default': 'None', 'annotation': 'float'}, {'name': 'crossover_rate', 'default': 'None', 'annotation': 'float'}, {'name': 'deap_verbose', 'default': 'None', 'annotation': 'bool'}]}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

