from skdecide.builders.discrete_optimization.generic_tools.do_problem import Problem, TypeObjective, TypeAttribute, Solution, \
    ObjectiveRegister, \
    ObjectiveHandling, EncodingRegister, ModeOptim
from typing import List, Union, Dict
import numpy as np
from skdecide.builders.discrete_optimization.generic_tools.graph_api import Graph


class ColoringSolution(Solution):
    def __init__(self, problem: Problem,
                 colors: Union[List[int], np.array] = None,
                 # colors_from0: Union[List[int], np.array] = None,
                 nb_color: int=None,
                 nb_violations: int = None):
        # assert(colors is not None or colors_from0 is not None)

        self.problem = problem
        self.colors = colors
        # self.colors_from0 = colors_from0
        self.nb_color = nb_color
        self.nb_violations = nb_violations

        # if self.colors is None:
        #     self.colors = self.problem.convert_list_from0_to_original_node_list(self.colors_from0)
        # if self.colors_from0 is None:
        #     self.colors_from0 = self.problem.convert_original_node_list_to_list_from0(self.colors)

    def copy(self):
        return ColoringSolution(problem=self.problem,
                                colors=list(self.colors),
                                # colors_from0=list(self.colors_from0),
                                nb_color=self.nb_color,
                                nb_violations=self.nb_violations)

    def lazy_copy(self):
        return ColoringSolution(problem=self.problem,
                                colors=self.colors,
                                # colors_from0=list(self.colors_from0),
                                nb_color=self.nb_color,
                                nb_violations=self.nb_violations)

    def to_reformated_solution(self):
        sol = ColoringSolution(problem=self.problem,
                               colors=transform_color_values_to_value_precede(self.colors),
                               nb_color=self.nb_color,
                               nb_violations=self.nb_violations)
        return sol

    def __str__(self):
        return "nb_color = "+str(self.nb_color)+"\n"+"nb_violations="+str(self.nb_violations)+"\n"+"colors="+str(self.colors)

    def change_problem(self, new_problem):
        self.__init__(problem=new_problem,
                      colors=list(self.colors),
                      # colors_from0=list(self.colors_from0),
                      nb_color=self.nb_color,
                      nb_violations=self.nb_violations)


def transform_color_values_to_value_precede(color_vector: List[int]):
    index_value_color = {}
    new_value = 0
    new_colors_vector = []
    for k in range(len(color_vector)):
        if color_vector[k] not in index_value_color:
            index_value_color[color_vector[k]] = new_value
            new_value += 1
        new_colors_vector += [index_value_color[color_vector[k]]]
    return new_colors_vector


class ColoringProblem(Problem):
    def __init__(self, graph: Graph):
        self.graph = graph
        self.number_of_nodes = len(self.graph.nodes_infos_dict)
        self.nodes_name = self.graph.nodes_name
        self.index_nodes_name = {self.nodes_name[i]: i for i in range(self.number_of_nodes)}
        self.index_to_nodes_name = {i: self.nodes_name[i] for i in range(self.number_of_nodes)}

        # self.list_from0_to_original_nodes = graph.nodes_name
        # self.original_nodes_name_to_list_from0 = {}
        # counter = 0
        # for i in graph.nodes_name:
        #     self.original_nodes_name_to_list_from0[i] = counter
        #     counter += 1

    def evaluate(self, variable: ColoringSolution) -> Dict[str, float]:
        if variable.nb_color is None:
            variable.nb_color = len(set(variable.colors))
            variable.nb_violations = self.count_violations(variable)
        return {"nb_colors": variable.nb_color, "nb_violations": variable.nb_violations}

    def satisfy(self, variable: ColoringSolution) -> bool:
        for e in self.graph.edges:
            if variable.colors[self.index_nodes_name[e[0]]] == \
                    variable.colors[self.index_nodes_name[e[1]]]:
                return False
        return True

    # TODO: We need to allow a user to change the arrity (or the range_value) of the encoding (test with GA)
    def get_attribute_register(self) -> EncodingRegister:
        dict_register = {
                        # "colors": {"type": [TypeAttribute.LIST_INTEGER],
                        #             "range_value": [0, self.number_of_nodes-1]},
                         "colors": {"name": "colors",
                                    "type": [TypeAttribute.LIST_INTEGER],
                                    "n": self.number_of_nodes,
                                    "arrity": self.number_of_nodes
                                    }
                         }
        return EncodingRegister(dict_register)

    def get_solution_type(self):
        return ColoringSolution

    # TODO: add maximzation or minimization to objective register (valid for any Problem)
    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {"nb_colors": {"type": TypeObjective.OBJECTIVE, "default_weight": -1},
                          "nb_violations": {"type": TypeObjective.PENALTY, "default_weight": -100}}
        return ObjectiveRegister(objective_sense=ModeOptim.MAXIMIZATION,
                                 objective_handling=ObjectiveHandling.AGGREGATE,
                                 dict_objective_to_doc=dict_objective)

    def get_dummy_solution(self):
        colors = list(range(self.number_of_nodes))
        solution = ColoringSolution(self, colors)
        self.evaluate(solution)
        return solution

    def count_violations(self, variable: ColoringSolution):
        val = 0
        for e in self.graph.edges:
            if variable.colors[self.index_nodes_name[e[0]]] == variable.colors[self.index_nodes_name[e[1]]]:
                val += 1
        return val

    # def convert_list_from0_to_original_node_list(self, list_from0):
    #     the_list = [self.list_from0_to_original_nodes[x] for x in list_from0]
    #     return the_list
    #
    # def convert_original_node_list_to_list_from0(self, the_list):
    #     list_from0 = [self.original_nodes_name_to_list_from0[i] for i in the_list]
    #     return list_from0

    def evaluate_from_encoding(self, int_vector, encoding_name):
        coloring_sol = None
        if encoding_name == 'colors':
            coloring_sol = ColoringSolution(problem=self,
                                            colors=int_vector
                                            # colors=self.convert_list_from0_to_original_node_list(int_vector)
                                            )
        elif encoding_name == 'custom':
            kwargs = {encoding_name: int_vector, 'problem': self}
            coloring_sol = ColoringSolution(**kwargs)

            # raise Exception("Decoding from custom encoding not implemented for coloring problem")
        # else:
        #     raise Exception('Encoding name not matching any suitable encoding for this problem - '
        #                     'Existing encoding for coloring problem are: ' +
        #                     str(self.get_attribute_register().dict_attribute_to_type.keys()))
        objectives = self.evaluate(coloring_sol)
        return objectives
