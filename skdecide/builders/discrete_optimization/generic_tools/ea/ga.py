from skdecide.builders.discrete_optimization.generic_tools.do_problem import Problem, EncodingRegister, TypeAttribute, ObjectiveHandling
from deap import creator, base, tools, algorithms
import random
from typing import Union, Optional, Any, Dict, List
import numpy as np
from enum import Enum
from skdecide.builders.discrete_optimization.generic_tools.ea.deap_wrappers import generic_mutate_wrapper
from skdecide.builders.discrete_optimization.generic_tools.do_mutation import Mutation
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from skdecide.builders.discrete_optimization.generic_tools.do_problem import build_evaluate_function_aggregated, ParamsObjectiveFunction, \
    ModeOptim, get_default_objective_setup, build_aggreg_function_and_params_objective

class DeapSelection(Enum):
    SEL_TOURNAMENT = 0
    SEL_RANDOM = 1
    SEL_BEST = 2
    SEL_ROULETTE = 4
    SEL_WORST = 5
    SEL_STOCHASTIC_UNIVERSAL_SAMPLING = 6


class DeapMutation(Enum):
    MUT_FLIP_BIT = 0  # bit
    MUT_SHUFFLE_INDEXES = 1  # perm
    MUT_UNIFORM_INT = 2  # int


class DeapCrossover(Enum):
    CX_UNIFORM = 0  # bit, int
    CX_UNIFORM_PARTIALY_MATCHED = 1  # perm
    CX_ORDERED = 2  # perm
    CX_ONE_POINT = 3  # bit, int
    CX_TWO_POINT = 4  # bit, int
    CX_PARTIALY_MATCHED = 5  # perm


class Ga():
    """Single objective GA

        Args:
            problem:
                the problem to solve
            encoding:
                name (str) of an encoding registered in the register solution of Problem
                or a dictionary of the form {'type': TypeAttribute, 'n': int} where type refers to a TypeAttribute and n
                 to the dimension of the problem in this encoding (e.g. length of the vector)
                by default, the first encoding in the problem register_solution will be used.

    """
    def __init__(self, problem: Problem,
                 mutation: Union[Mutation, DeapMutation] = None,
                 crossover: DeapCrossover = None,
                 selection: DeapSelection = None,
                 encoding: Optional[Union[str, Dict[str, Any]]] = None,
                 objective_handling: Optional[ObjectiveHandling] = None,
                 objectives: Optional[Union[str,List[str]]] = None,
                 objective_weights: Optional[List[float]] = None,
                 pop_size: int = None,
                 max_evals: int = None,
                 mut_rate: float = None,
                 crossover_rate: float = None,
                 tournament_size: float = None,
                 deap_verbose: bool = None
                 ):



        self._default_crossovers = {TypeAttribute.LIST_BOOLEAN: DeapCrossover.CX_UNIFORM,
                                    TypeAttribute.LIST_INTEGER: DeapCrossover.CX_ONE_POINT,
                                    TypeAttribute.LIST_INTEGER_SPECIFIC_ARRITY: DeapCrossover.CX_ONE_POINT,
                                    TypeAttribute.PERMUTATION: DeapCrossover.CX_UNIFORM_PARTIALY_MATCHED}
        self._default_mutations = {TypeAttribute.LIST_BOOLEAN: DeapMutation.MUT_FLIP_BIT,
                                   TypeAttribute.LIST_INTEGER: DeapMutation.MUT_UNIFORM_INT,
                                   TypeAttribute.LIST_INTEGER_SPECIFIC_ARRITY: DeapMutation.MUT_UNIFORM_INT,
                                   TypeAttribute.PERMUTATION: DeapMutation.MUT_SHUFFLE_INDEXES}
        self._default_selection = DeapSelection.SEL_TOURNAMENT

        self.problem = problem
        if pop_size is not None:
            self._pop_size = pop_size
        else:
            self._pop_size = 100

        if max_evals is not None:
            self._max_evals = max_evals
        else:
            self._max_evals = 100 * self._pop_size
            print('No value specified for max_evals. Using the default 10*pop_size - This should really be set carefully')

        if mut_rate is not None:
            self._mut_rate = mut_rate
        else:
            self._mut_rate = 0.1

        if crossover_rate is not None:
            self._crossover_rate = crossover_rate
        else:
            self._crossover_rate = 0.9

        self.problem = problem

        if tournament_size is not None:
            self._tournament_size = tournament_size
        else:
            self._tournament_size = 0.2 # as a percentage of the population

        if deap_verbose is not None:
            self._deap_verbose = deap_verbose
        else:
            self._deap_verbose = True

        # set encoding
        register_solution: EncodingRegister = problem.get_attribute_register()
        self._encoding_name = None
        self._encoding_variable_name = None
        if encoding is not None and isinstance(encoding, str):
            # check name specified is in problem register
            print(encoding)
            if encoding in register_solution.dict_attribute_to_type.keys():
                self._encoding_name = encoding
                self._encoding_variable_name = register_solution.dict_attribute_to_type[self._encoding_name]['name']
                self._encoding_type = register_solution.dict_attribute_to_type[self._encoding_name]['type'][0]
                self.n = register_solution.dict_attribute_to_type[self._encoding_name]['n']

                if self._encoding_type == TypeAttribute.LIST_INTEGER:
                    self.arrity = register_solution.dict_attribute_to_type[self._encoding_name]['arrity']
                    self.arrities = [self.arrity for i in range(self.n)]
                else:
                    self.arrity = None
                if self._encoding_type == TypeAttribute.LIST_INTEGER_SPECIFIC_ARRITY:
                    self.arrities = register_solution.dict_attribute_to_type[self._encoding_name]['arrities']
                # else:
                #     self.arrities = None

        if encoding is not None and isinstance(encoding, Dict):
            # check there is a type key and a n key
            if 'name' in encoding.keys() and 'type' in encoding.keys() and 'n' in encoding.keys():
                self._encoding_name = "custom"
                self._encoding_variable_name = encoding['name']
                self._encoding_type = encoding['type'][0]
                self.n = encoding['n']
                if 'arrity' in encoding.keys():
                    self.arrity = encoding['arrity']
                    self.arrities = [self.arrity for i in range(self.n)]
                if 'arrities' in encoding.keys():
                    self.arrities = register_solution.dict_attribute_to_type[self._encoding_name]['arrities']
            else:
                print('Erroneous encoding provided as input (encoding name not matching encoding of problem or custom '
                      'definition not respecting encoding dict entry format, trying to use default one instead')

        if self._encoding_name is None:
            if len(register_solution.dict_attribute_to_type.keys()) == 0:
                raise Exception("An encoding of type TypeAttribute should be specified or at least 1 TypeAttribute "
                                "should be defined in the RegisterSolution of your Problem")
            print(register_solution.dict_attribute_to_type)
            print(register_solution.dict_attribute_to_type.keys())
            self._encoding_name = list(register_solution.dict_attribute_to_type.keys())[0]
            self._encoding_variable_name = register_solution.dict_attribute_to_type[self._encoding_name]['name']
            self._encoding_type = register_solution.dict_attribute_to_type[self._encoding_name]['type'][0] # TODO : while it's usually a list we could also have a unique value(not a list)
            self.n = register_solution.dict_attribute_to_type[self._encoding_name]['n']

            if self._encoding_type == TypeAttribute.LIST_INTEGER:
                self.arrity = register_solution.dict_attribute_to_type[self._encoding_name]['arrity']
                self.arrities = [self.arrity for i in range(self.n)]
            else:
                self.arrity = None
            if self._encoding_type == TypeAttribute.LIST_INTEGER_SPECIFIC_ARRITY:
                self.arrities = register_solution.dict_attribute_to_type[self._encoding_name]['arrities']
            # else:
            #     self.arrities = None


        if self._encoding_type == TypeAttribute.LIST_BOOLEAN:
            self.arrity = 2
            self.arities = [2 for i in range(self.n)]

        print("Encoding used by the GA: "+self._encoding_name+": "+str(self._encoding_type)+" of length "+str(self.n))

        # set objective handling stuff
        if objective_handling is None:
            self._objective_handling = ObjectiveHandling.SINGLE
        else:
            self._objective_handling = objective_handling

        self._objectives = objectives
        if (isinstance(self._objectives, List) and len(self._objectives) > 1) \
                and self._objective_handling == ObjectiveHandling.SINGLE:
            print('Many objectives specified but single objective handling, using the first objective in the dictionary')

        self._objective_weights = objective_weights
        if (self._objective_weights is None) \
                or self._objective_weights is not None \
                and((len(self._objective_weights) != len(self._objectives)
                     and self._objective_handling == ObjectiveHandling.AGGREGATE)):
            print('Objective weight issue: no weight given or size of weights and objectives lists mismatch. '
                  'Setting all weights to default 1 value.')
            self._objective_weights = [1 for i in range(len(self._objectives))]

        if selection is None:
            self._selection_type = self._default_selection
        else:
            self._selection_type = selection

        # DEAP toolbox setup
        self._toolbox = base.Toolbox()

        # Define representation
        creator.create(
            "fitness", base.Fitness, weights=(1.0,) # we keep this to 1 and let the user provides the weights for each subobjective
        )  # (a negative weight defines the objective as a minimisation)
        creator.create(
            "individual", list, fitness=creator.fitness
        )  # associate the fitness function to the individual type

        # Create the individuals required by the encoding
        if self._encoding_type == TypeAttribute.LIST_BOOLEAN:
            self._toolbox.register("bit", random.randint, 0, 1)  # Each element of a solution is a bit (i.e. an int between 0 and 1 incl.)

            self._toolbox.register(
                "individual", tools.initRepeat, creator.individual, self._toolbox.bit, n=self.n
            )  # An individual (aka solution) contains n bits
        elif self._encoding_type == TypeAttribute.PERMUTATION:
            self._toolbox.register("permutation_indices", random.sample, range(self.n), self.n)
            self._toolbox.register("individual", tools.initIterate, creator.individual,
                                   self._toolbox.permutation_indices)
        elif self._encoding_type == TypeAttribute.LIST_INTEGER:
            self._toolbox.register("int_val", random.randint, 0, self.arrity-1)
            self._toolbox.register("individual", tools.initRepeat, creator.individual, self._toolbox.int_val, n=self.n)
        elif self._encoding_type == TypeAttribute.LIST_INTEGER_SPECIFIC_ARRITY:
            gen_idx = lambda: [random.randint(0, arrity-1) for arrity in self.arrities]
            self._toolbox.register("individual", tools.initIterate, creator.individual, gen_idx)

        self._toolbox.register(
            "population", tools.initRepeat, list, self._toolbox.individual, n=self._pop_size
        )  # A population is made of pop_size individuals

        # Define objective function
        self._toolbox.register(
            "evaluate",
            self.evaluate_problem,
        )

        # Define crossover
        if crossover is None:
            self._crossover = self._default_crossovers[self._encoding_type]
        else:
            self._crossover = crossover

        # if self._encoding_type == TypeAttribute.LIST_BOOLEAN:
        if self._crossover == DeapCrossover.CX_UNIFORM:
            self._toolbox.register("mate", tools.cxUniform, indpb=self._crossover_rate)
        elif self._crossover == DeapCrossover.CX_ONE_POINT:
            self._toolbox.register("mate", tools.cxOnePoint)
        elif self._crossover == DeapCrossover.CX_TWO_POINT:
                self._toolbox.register("mate", tools.cxTwoPoint)
    # elif self._encoding_type == TypeAttribute.PERMUTATION:
        elif self._crossover == DeapCrossover.CX_UNIFORM_PARTIALY_MATCHED:
            self._toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=0.5)
        elif self._crossover == DeapCrossover.CX_ORDERED:
            self._toolbox.register("mate", tools.cxOrdered)
        elif self._crossover == DeapCrossover.CX_PARTIALY_MATCHED:
            self._toolbox.register("mate", tools.cxPartialyMatched)
        else:
            print("Crossover of specified type not handled!")

        # Define mutation
        if mutation is None:
            self._mutation = self._default_mutations[self._encoding_type]
        else:
            self._mutation = mutation

        if isinstance(self._mutation, Mutation):
            self._toolbox.register("mutate", generic_mutate_wrapper, problem=self.problem, encoding_name=self._encoding_variable_name, indpb=self._mut_rate,
                                   solution_fn=self.problem.get_solution_type(), custom_mutation=mutation)
        elif isinstance(self._mutation, DeapMutation):
            if self._mutation == DeapMutation.MUT_FLIP_BIT:
                self._toolbox.register("mutate", tools.mutFlipBit, indpb=self._mut_rate)  # Choice of mutation operator
            elif self._mutation == DeapMutation.MUT_SHUFFLE_INDEXES:
                self._toolbox.register("mutate", tools.mutShuffleIndexes, indpb=self._mut_rate)  # Choice of mutation operator
            elif self._mutation == DeapMutation.MUT_UNIFORM_INT:
                # print('DEAP-GA - self.arrities: ', self.arrities)
                # print('DEAP-GA - self.arrity: ', self.arrity)

                # self._toolbox.register("mutate", tools.mutUniformInt, low=0, up=self.arrity-1, indpb=self._mut_rate)
                self._toolbox.register("mutate", tools.mutUniformInt, low=0, up=self.arrities, indpb=self._mut_rate)

        # Choice of selection
        if self._selection_type == DeapSelection.SEL_TOURNAMENT:
            self._toolbox.register("select", tools.selTournament, tournsize=int(self._tournament_size*self._pop_size))
        elif self._selection_type == DeapSelection.SEL_RANDOM:
            self._toolbox.register("select", tools.selRandom)
        elif self._selection_type == DeapSelection.SEL_BEST:
            self._toolbox.register("select", tools.selBest)
        elif self._selection_type == DeapSelection.SEL_ROULETTE:
            self._toolbox.register("select", tools.selRoulette)
        elif self._selection_type == DeapSelection.SEL_WORST:
            self._toolbox.register("select", tools.selWorst)
        elif self._selection_type == DeapSelection.SEL_STOCHASTIC_UNIVERSAL_SAMPLING:
            self._toolbox.register("select", tools.selStochasticUniversalSampling)

        self.aggreg_from_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.problem,
                                                       params_objective_function=
                                                       None)  # TODO: That should probably be set to somthing else than None.

    def evaluate_problem(self, int_vector):
        # encoding_name = self._encoding_name
        objective_values = self.problem.evaluate_from_encoding(int_vector, self._encoding_variable_name)
        if self._objective_handling == ObjectiveHandling.SINGLE:
            if (self._objectives is None) or (self._objectives[0] not in list(objective_values.keys())):
                default_key = list(objective_values.keys())[0]
                val = self._objectives[0] * objective_values[default_key]
            else:
                val = objective_values[self._objectives[0]]
        elif self._objective_handling == 'aggregate':
            val = sum([objective_values[self._objectives[i]] * self._objective_weights[i]
                       for i in range(len(self._objectives))])
        elif self._objective_handling == ObjectiveHandling.AGGREGATE:
            val = sum([objective_values[self._objectives[i]] * self._objective_weights[i] for i in range(len(self._objectives))])
        return (val,)

    def solve(self, **kwargs):
        # Initialise the population (here at random)
        population = self._toolbox.population()

        fits = self._toolbox.map(self._toolbox.evaluate, population)
        for fit, ind in zip(fits, population):
            ind.fitness.values = fit

        #  Define the statistics to collect at each generation
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run the GA: final population and statistics logbook are created
        pop_vector, logbook = algorithms.eaSimple(
            population=population,
            toolbox=self._toolbox,
            cxpb=self._crossover_rate,
            mutpb=self._mut_rate,
            ngen=int(self._max_evals / self._pop_size),
            stats=stats,
            halloffame=hof,
            verbose=self._deap_verbose,
        )

        best_vector = hof[0]

        s_pure_int = [i for i in best_vector]
        kwargs = {self._encoding_variable_name: s_pure_int, 'problem': self.problem}
        problem_sol = self.problem.get_solution_type()(**kwargs)

        result_storage = ResultStorage(list_solution_fits=[(problem_sol,
                                                            self.aggreg_from_sol(problem_sol))],
                                       best_solution=problem_sol,
                                       mode_optim=self.params_objective_function.sense_function)
        return result_storage

        # return problem_sol




