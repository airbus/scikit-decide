# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Callable, Set, Tuple

from skdecide import Domain, Solver
from skdecide.core import Value
from skdecide.builders.domain import MultiAgent, SingleAgent, Sequential
from skdecide.builders.solver import DeterministicPolicies, Utilities


# TODO: remove Markovian req?
class D(Domain, MultiAgent, Sequential):
    pass


class MAHD(Solver, DeterministicPolicies, Utilities):
    T_domain = D
    
    def __init__(self,
                    multiagent_solver_class,
                    singleagent_solver_class,
                    multiagent_domain_class,
                    singleagent_domain_class,
                    multiagent_domain_factory: Callable[[], Domain] = None,
                    singleagent_domain_factory: Callable[[Domain, Any], Domain] = None,
                    multiagent_solver_kwargs = None,
                    singleagent_solver_kwargs = None) -> None:
        if multiagent_solver_kwargs is None:
            multiagent_solver_kwargs = {}
        if 'heuristic' in multiagent_solver_kwargs:
            print('\x1b[3;33;40m' + 'Multi-agent solver heuristic will be overwritten by MAHD!' + '\x1b[0m')
        multiagent_solver_kwargs['heuristic'] = lambda d, o: self._multiagent_heuristic(o)
        self._multiagent_solver = multiagent_solver_class(**multiagent_solver_kwargs)
        self._multiagent_domain_class = multiagent_domain_class
        self._multiagent_domain_factory = multiagent_domain_factory
        self._multiagent_domain = self._multiagent_domain_factory()

        self._singleagent_solver_class = singleagent_solver_class
        self._singleagent_solver_kwargs = singleagent_solver_kwargs
        self._singleagent_domain_class = singleagent_domain_class
        self._singleagent_domain_factory = singleagent_domain_factory
        
        self._singleagent_domains = {}
        self._singleagent_solvers = {}
        if self._singleagent_solver_kwargs is None:
            self._singleagent_solver_kwargs = {}
        
        for a in self._multiagent_domain.get_agents():
            self._singleagent_solvers[a] = self._singleagent_solver_class(
                                                **self._singleagent_solver_kwargs)
            self._singleagent_domain_class.solve_with(
                                                solver=self._singleagent_solvers[a],
                                                domain_factory=lambda: self._singleagent_domain_factory(self._multiagent_domain, a) if self._singleagent_domain_factory is not None else None)
        
        self._singleagent_solutions = {a: {} for a in self._multiagent_domain.get_agents()}

    def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
        self._multiagent_domain_class.solve_with(solver=self._multiagent_solver, 
                                                 domain_factory=domain_factory)
    
    def _get_next_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
        return self._multiagent_solver._get_next_action(observation)

    def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
        return self._multiagent_solver._get_utility(observation)
    
    def _multiagent_heuristic(self, observation: D.T_agent[D.T_observation]) -> Tuple[D.T_agent[Value[D.T_value]], D.T_agent[D.T_concurrency[D.T_event]]]:
        h = {}
        for a, s in self._singleagent_solvers.items():
            if observation[a] not in self._singleagent_solutions[a]:
                undefined_solution = False
                s.solve_from(observation[a])
                if hasattr(self._singleagent_solvers[a], 'get_policy'):
                    p = self._singleagent_solvers[a].get_policy()
                    for ps, pav in p.items():
                        self._singleagent_solutions[a][ps] = pav[::-1]
                    undefined_solution = (observation[a] not in self._singleagent_solutions[a])
                else:
                    if not s.is_solution_defined_for(observation[a]):
                        undefined_solution = True
                    else:
                        self._singleagent_solutions[a][observation[a]] = (s.get_utility(observation[a]),
                                                                          s.get_next_action(observation[a]))
                if undefined_solution:
                    is_terminal = ((hasattr(self._get_singleagent_domain(a), 'is_goal') and
                                    self._get_singleagent_domain(a).is_goal(observation[a])) or
                                   (hasattr(self._get_singleagent_domain(a), 'is_terminal') and
                                    self._get_singleagent_domain(a).is_terminal(observation[a])))
                    if not is_terminal:
                        print('\x1b[3;33;40m' +
                            '/!\ Solution not defined for agent {} in non terminal state {}'.format(a, observation[a]) +
                            ': Assigning default action! (is it a terminal state without no-op action?)'
                            '\x1b[0m')
                    try:
                        self._singleagent_solutions[a][observation[a]] = (
                            0, self._get_singleagent_domain(a).get_applicable_actions(observation[a]).sample()
                        )
                    except Exception as err:
                        terminal_str = 'terminal ' if is_terminal else ''
                        raise RuntimeError('Cannot sample applicable action '\
                                           'for agent {} in {}state {} ' \
                                           '(original exception is: {})'.format(
                                           a, terminal_str, observation[a], err))
        if issubclass(self._multiagent_solver.T_domain, SingleAgent):
            h = (Value(cost=sum(p[observation[a]][0] for a, p in self._singleagent_solutions.items())),
                 {a: p[observation[a]][1] for a, p in self._singleagent_solutions.items()})
        else:
            h = ({a: Value(cost=p[observation[a]][0]) for a, p in self._singleagent_solutions.items()},
                 {a: p[observation[a]][1] for a, p in self._singleagent_solutions.items()})
        return h
    
    def _get_singleagent_domain(self, agent):
        if agent not in self._singleagent_domains:
            self._singleagent_domains[agent] = self._singleagent_domain_factory(self._multiagent_domain, agent)
        return self._singleagent_domains[agent]
    
    def _initialize(self):
        self._multiagent_solver._initialize()
        for a, s in self._singleagent_solvers.items():
            s._initialize()
    
    def _cleanup(self):
        self._multiagent_solver._cleanup()
        for a, s in self._singleagent_solvers.items():
            s._cleanup()
