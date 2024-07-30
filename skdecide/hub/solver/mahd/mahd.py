# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
from typing import Any, Callable, List, Optional, Set, Tuple, Type

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    SubBrickHyperparameter,
    SubBrickKwargsHyperparameter,
)

from skdecide import Domain, Solver
from skdecide.builders.domain import MultiAgent, Sequential, SingleAgent
from skdecide.builders.solver import DeterministicPolicies, FromAnyState, Utilities
from skdecide.core import Value


# TODO: remove Markovian req?
class D(Domain, MultiAgent, Sequential):
    pass


class MAHD(Solver, DeterministicPolicies, Utilities, FromAnyState):
    """This is an experimental implementation of a centralized multi-agent heuristic solver
    which makes use of a master heuristic-search solver (i.e. MARTDP, HMCTS, HUCT) to solve
    the centralized multi-agent domain by calling a heuristic sub-solver (i.e. LRTDP, ILAO*,
    A*) independently on each agent's domain to compute the heuristic
    estimates of each agent as if it involved alone in the environment."""

    T_domain = D

    hyperparameters = [
        SubBrickHyperparameter(name="multiagent_solver_class", choices=[]),
        SubBrickKwargsHyperparameter(
            name="multiagent_solver_kwargs",
            subbrick_hyperparameter="multiagent_solver_class",
        ),
        SubBrickHyperparameter(name="singleagent_solver_class", choices=[]),
        SubBrickKwargsHyperparameter(
            name="singleagent_solver_kwargs",
            subbrick_hyperparameter="singleagent_solver_class",
        ),
    ]

    def __init__(
        self,
        multiagent_solver_class: Type[Solver],
        singleagent_solver_class: Type[Solver],
        multiagent_domain_factory: Callable[[], Domain],
        singleagent_domain_class: Optional[Type[Domain]] = None,
        singleagent_domain_factory: Optional[Callable[[Domain, Any], Domain]] = None,
        multiagent_solver_kwargs=None,
        singleagent_solver_kwargs=None,
        callback: Callable[[MAHD], bool] = lambda solver: False,
    ) -> None:
        """Construct a MAHD solver instance

        # Parameters
        multiagent_solver_class (Type[Solver]): Class type of the higher-level multi-agent solver
            called on the main multi-agent domain
        singleagent_solver_class (Type[Solver]): Class type of the lower-level single-agent solver
            called on the single-agent domain used to compute single agent heuristic estimates
        multiagent_domain_factory (Callable[[], Domain]): Lambda function called to create a
            multi-agent domain instance
        singleagent_domain_class (Optional[Type[Domain]], optional): Class type of the single-agent
            domain used to compute single agent heuristic estimates. Defaults to None.
        singleagent_domain_factory (Optional[Callable[[Domain, Any], Domain]], optional): Lambda function
            which takes as arguments the multi-agent domain and one agent, and that returns a domain
            instance for this single agent; it is called to create the single-agent domain used
            to compute single agent heuristic estimates. Defaults to None.
        multiagent_solver_kwargs (_type_, optional): Optional arguments to be passed to the higher-level
            multi-agent solver. Defaults to None.
        singleagent_solver_kwargs (_type_, optional): Optional arguments to be passed to the lower-level
            single-agent solver. Defaults to None.
        callback: function called at each solver iteration. If returning true, the solve process stops.

        !!! warning
            One of `singleagent_domain_class` or `singleagent_domain_factory` must be not None, otherwise
            a `ValueError` exception is raised.

        """
        Solver.__init__(self, domain_factory=multiagent_domain_factory)
        self.callback = callback
        if multiagent_solver_kwargs is None:
            multiagent_solver_kwargs = {}
        if "heuristic" in multiagent_solver_kwargs:
            print(
                "\x1b[3;33;40m"
                + "Multi-agent solver heuristic will be overwritten by MAHD!"
                + "\x1b[0m"
            )
        multiagent_solver_kwargs["heuristic"] = lambda d, o: self._multiagent_heuristic(
            o
        )
        if ("domain_factory" not in multiagent_solver_kwargs) and (
            "domain_factory"
            in inspect.signature(multiagent_solver_class.__init__).parameters
        ):
            multiagent_solver_kwargs["domain_factory"] = multiagent_domain_factory
        # add callback to multiagent solver
        if "callback" in inspect.signature(multiagent_solver_class.__init__).parameters:
            mahd_callback = MahdCallback(solver=self, callback=callback)
            if "callback" in multiagent_solver_kwargs:
                callbacks = [mahd_callback, multiagent_solver_kwargs["callback"]]
                multiagent_solver_kwargs["callback"] = CallbackList(callbacks=callbacks)
            else:
                multiagent_solver_kwargs["callback"] = mahd_callback

        self._multiagent_solver = multiagent_solver_class(**multiagent_solver_kwargs)
        self._multiagent_domain_factory = multiagent_domain_factory
        self._multiagent_domain = self._multiagent_domain_factory()

        self._singleagent_solver_class = singleagent_solver_class
        self._singleagent_solver_kwargs = singleagent_solver_kwargs
        if singleagent_domain_factory is None:
            if singleagent_domain_class is None:
                raise ValueError(
                    "singleagent_domain_factory and singleagent_domain_class cannot be None together."
                )
            else:
                self._singleagent_domain_factory = (
                    lambda multiagent_domain, agent: singleagent_domain_class()
                )
        else:
            self._singleagent_domain_factory = singleagent_domain_factory

        self._singleagent_domains = {}
        self._singleagent_solvers = {}
        if self._singleagent_solver_kwargs is None:
            self._singleagent_solver_kwargs = {}

        for a in self._multiagent_domain.get_agents():
            singleagent_solver_kwargs = dict(self._singleagent_solver_kwargs)
            singleagent_solver_kwargs[
                "domain_factory"
            ] = lambda: self._singleagent_domain_factory(self._multiagent_domain, a)
            self._singleagent_solvers[a] = self._singleagent_solver_class(
                **singleagent_solver_kwargs
            )

        self._singleagent_solutions = {
            a: {} for a in self._multiagent_domain.get_agents()
        }

    def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
        """Run the higher-level multi-agent heuristic solver from a given joint state

        # Parameters
        memory (D.T_memory[D.T_state]): Joint state from which to run the MAHD algorithm
        """
        self._multiagent_solver._solve_from(
            memory=memory,
        )

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        """Gets the best computed joint action according to the higher-level heuristic
            multi-agent solver in a given joint state.

        # Parameters
        observation (D.T_agent[D.T_observation]): Joint state for which the best action
            is requested

        # Returns
        D.T_agent[D.T_concurrency[D.T_event]]: Best computed joint action
        """
        return self._multiagent_solver._get_next_action(observation)

    def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
        """Gets the best value in a given joint state according to the higher-level
        heuristic multi-agent solver

        # Parameters
        observation (D.T_agent[D.T_observation]): Joint state from which the best value
                is requested

        # Returns
        D.T_value: _description_
        """
        return self._multiagent_solver._get_utility(observation)

    def _multiagent_heuristic(
        self, observation: D.T_agent[D.T_observation]
    ) -> Tuple[D.T_agent[Value[D.T_value]], D.T_agent[D.T_concurrency[D.T_event]]]:
        """Computes the multi-agent relaxed heuristics to be used by the higher-level
            multi-agent solver as a pair of 2 dictionaries: one from single agents to
            their individual heuristic estimates, and one from single agents to their
            heuristic best actions in the given joint state

        # Parameters
        observation (D.T_agent[D.T_observation]): Joint state from which the relaxed
            multi-agent heuristics are computed

        !!! warning
            Throws a `RuntimeError` exception if the single-agent solver cannot compute a
            heuristic action for one of the agents and that no applicable action can be
            then sampled for this agent

        # Returns
        Tuple[D.T_agent[Value[D.T_value]], D.T_agent[D.T_concurrency[D.T_event]]]:
            Pair of 2 dictionaries: one from single agents to their individual
            heuristic estimates, and one from single agents to their
            heuristic best actions in the given joint state
        """
        h = {}
        for a, s in self._singleagent_solvers.items():
            if observation[a] not in self._singleagent_solutions[a]:
                undefined_solution = False
                s.solve_from(observation[a])
                if hasattr(self._singleagent_solvers[a], "get_policy"):
                    p = self._singleagent_solvers[a].get_policy()
                    for ps, pav in p.items():
                        self._singleagent_solutions[a][ps] = pav[::-1]
                    undefined_solution = (
                        observation[a] not in self._singleagent_solutions[a]
                    )
                else:
                    if not s.is_solution_defined_for(observation[a]):
                        undefined_solution = True
                    else:
                        self._singleagent_solutions[a][observation[a]] = (
                            s.get_utility(observation[a]),
                            s.get_next_action(observation[a]),
                        )
                if undefined_solution:
                    is_terminal = (
                        hasattr(self._get_singleagent_domain(a), "is_goal")
                        and self._get_singleagent_domain(a).is_goal(observation[a])
                    ) or (
                        hasattr(self._get_singleagent_domain(a), "is_terminal")
                        and self._get_singleagent_domain(a).is_terminal(observation[a])
                    )
                    if not is_terminal:
                        print(
                            "\x1b[3;33;40m"
                            + "/!\ Solution not defined for agent {} in non terminal state {}".format(
                                a, observation[a]
                            )
                            + ": Assigning default action! (is it a terminal state without no-op action?)"
                            "\x1b[0m"
                        )
                    try:
                        self._singleagent_solutions[a][observation[a]] = (
                            Value(cost=0),
                            self._get_singleagent_domain(a)
                            .get_applicable_actions(observation[a])
                            .sample(),
                        )
                    except Exception as err:
                        terminal_str = "terminal " if is_terminal else ""
                        raise RuntimeError(
                            "Cannot sample applicable action "
                            "for agent {} in {}state {} "
                            "(original exception is: {})".format(
                                a, terminal_str, observation[a], err
                            )
                        )
        if issubclass(self._multiagent_solver.T_domain, SingleAgent):
            h = (
                Value(
                    cost=sum(
                        p[observation[a]][0].cost
                        for a, p in self._singleagent_solutions.items()
                    )
                ),
                {
                    a: p[observation[a]][1]
                    for a, p in self._singleagent_solutions.items()
                },
            )
        else:
            h = (
                {
                    a: Value(cost=p[observation[a]][0].cost)
                    for a, p in self._singleagent_solutions.items()
                },
                {
                    a: p[observation[a]][1]
                    for a, p in self._singleagent_solutions.items()
                },
            )
        return h

    def _get_singleagent_domain(self, agent):
        """Gets the single-agent domain of a given agent, potentially
            building it from the single-agent domain factory given in the
            MAHD instance's constructor if it has not been yet created for
            this agent

        # Parameters
            agent (_type_): Agent for which the single-agent domain is requested

        # Returns
            _type_: Single-agent domain instance
        """
        if agent not in self._singleagent_domains:
            self._singleagent_domains[agent] = self._singleagent_domain_factory(
                self._multiagent_domain, agent
            )
        return self._singleagent_domains[agent]

    def _initialize(self):
        """Initializes the higher-level multi-agent solver and each lower-level
        single-agent solver
        """
        self._multiagent_solver._initialize()
        for a, s in self._singleagent_solvers.items():
            s._initialize()

    def _cleanup(self):
        """Cleans up the higher-level multi-agent solver and each lower-level
        single-agent solver
        """
        self._multiagent_solver._cleanup()
        for a, s in self._singleagent_solvers.items():
            s._cleanup()


class MahdCallback:
    def __init__(self, solver: MAHD, callback: Callable[[MAHD], bool]):
        self.callback = callback
        self.solver = solver

    def __call__(self, solver: Solver, *args, **kwargs):
        return self.callback(self.solver)


class CallbackList:
    def __init__(self, callbacks: List[Callable[[...], bool]]):
        self.callbacks = callbacks

    def __call__(self, *args, **kwargs):
        stopping = False
        for callback in self.callbacks:
            stopping = stopping or callback(*args, **kwargs)
        return stopping
