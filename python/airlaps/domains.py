# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This module contains base classes for quickly building domains."""
from __future__ import annotations

import os
from typing import NewType, Optional, Callable
from multiprocessing import Pool, Manager
from multiprocessing.managers import SyncManager
from queue import LifoQueue

from airlaps.core import autocast_all
from airlaps.builders.domain.agent import MultiAgent, SingleAgent
from airlaps.builders.domain.concurrency import Parallel, Sequential
from airlaps.builders.domain.dynamics import Environment, Simulation, EnumerableTransitions, DeterministicTransitions
from airlaps.builders.domain.events import Events, Actions
from airlaps.builders.domain.goals import Goals
from airlaps.builders.domain.initialization import Initializable, UncertainInitialized, DeterministicInitialized
from airlaps.builders.domain.memory import History, Markovian
from airlaps.builders.domain.observability import PartiallyObservable, TransformedObservable, FullyObservable
from airlaps.builders.domain.value import Rewards, PositiveCosts
if False:  # trick to avoid circular import & IDE error ("Unresolved reference 'Solver'")
    from airlaps.solvers import Solver

__all__ = ['Domain', 'RLDomain', 'MultiAgentRLDomain', 'StatelessSimulatorDomain', 'MDPDomain', 'POMDPDomain',
           'GoalMDPDomain', 'GoalPOMDPDomain', 'DeterministicPlanningDomain']


# MAIN BASE CLASS

class Domain(MultiAgent, Parallel, Environment, Events, History, PartiallyObservable, Rewards):
    """This is the highest level domain class (inheriting top-level class for each mandatory domain characteristic).

    This helper class can be used as the main base class for domains.

    Typical use:
    ```python
    class D(Domain, ...)
    ```

    with "..." replaced when needed by a number of classes from following domain characteristics (the ones in
    parentheses are optional):

    - **agent**: MultiAgent -> SingleAgent
    - **concurrency**: Parallel -> Sequential
    - **(constraints)**: Constrained
    - **dynamics**: Environment -> Simulation -> UncertainTransitions -> EnumerableTransitions
      -> DeterministicTransitions
    - **events**: Events -> Actions -> UnrestrictedActions
    - **(goals)**: Goals
    - **(initialization)**: Initializable -> UncertainInitialized -> DeterministicInitialized
    - **memory**: History -> FiniteHistory -> Markovian -> Memoryless
    - **observability**: PartiallyObservable -> TransformedObservable -> FullyObservable
    - **(renderability)**: Renderable
    - **value**: Rewards -> PositiveCosts
    """
    T_state = NewType('T_state', object)
    T_observation = NewType('T_observation', object)
    T_event = NewType('T_event', object)
    T_value = NewType('T_value', object)
    T_info = NewType('T_info', object)

    @classmethod
    def solve_with(cls, solver_factory: Callable[[], Solver],
                   domain_factory: Optional[Callable[[], Domain]] = None, load_path: Optional[str] = None) -> Solver:
        """Solve the domain with a new or loaded solver and return it auto-cast to the level of the domain.

        By default, #Solver.check_domain() provides some boilerplate code and internally
        calls #Solver._check_domain_additional() (which returns True by default but can be overridden  to define
        specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
        domain requirements are met.

        # Parameters
        solver_factory: A callable with no argument returning the new solver (can be just a solver class).
        domain_factory: A callable with no argument returning the domain to solve (factory is the domain class if None).
        load_path: The path to restore the solver state from (if None, the solving process will be launched instead).

        # Returns
        The new solver (auto-cast to the level of the domain).
        """
        if domain_factory is None:
            domain_factory = cls
        solver = solver_factory()
        if load_path is not None:

            # TODO: avoid repeating this code somehow (identical in solver.solve(...))? Is factory necessary (vs cls)?
            def cast_domain_factory():
                domain = domain_factory()
                autocast_all(domain, domain, solver.T_domain)
                return domain

            solver.load(load_path, cast_domain_factory)
        else:
            solver.solve(domain_factory)
        autocast_all(solver, solver.T_domain, cls)
        return solver


# ALTERNATE BASE CLASSES (for typical combinations)

class RLDomain(Domain, SingleAgent, Sequential, Environment, Actions, Initializable, Markovian, TransformedObservable,
               Rewards):
    """This is a typical Reinforcement Learning domain class.

    This helper class can be used as an alternate base class for domains, inheriting the following:

    - Domain
    - SingleAgent
    - Sequential
    - Environment
    - Actions
    - Initializable
    - Markovian
    - TransformedObservable
    - Rewards

    Typical use:
    ```python
    class D(RLDomain)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class D(RLDomain, FullyObservable)
        ```
    """
    pass


class MultiAgentRLDomain(Domain, MultiAgent, Sequential, Environment, Actions, Initializable, Markovian,
                         TransformedObservable, Rewards):
    """This is a typical multi-agent Reinforcement Learning domain class.

    This helper class can be used as an alternate base class for domains, inheriting the following:

    - Domain
    - MultiAgent
    - Sequential
    - Environment
    - Actions
    - Initializable
    - Markovian
    - TransformedObservable
    - Rewards

    Typical use:
    ```python
    class D(RLDomain)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class D(RLDomain, FullyObservable)
        ```
    """
    pass


class StatelessSimulatorDomain(Domain, SingleAgent, Sequential, Simulation, Actions, Markovian, TransformedObservable,
                               Rewards):
    """This is a typical stateless simulator domain class.

    This helper class can be used as an alternate base class for domains, inheriting the following:

    - Domain
    - SingleAgent
    - Sequential
    - Simulation
    - Actions
    - Markovian
    - TransformedObservable
    - Rewards

    Typical use:
    ```python
    class D(StatelessSimulatorDomain)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class D(RLDomain, FullyObservable)
        ```
    """
    pass


class MDPDomain(Domain, SingleAgent, Sequential, EnumerableTransitions, Actions, DeterministicInitialized, Markovian,
                FullyObservable, Rewards):
    """This is a typical Markov Decision Process domain class.

    This helper class can be used as an alternate base class for domains, inheriting the following:

    - Domain
    - SingleAgent
    - Sequential
    - EnumerableTransitions
    - Actions
    - DeterministicInitialized
    - Markovian
    - FullyObservable
    - Rewards

    Typical use:
    ```python
    class D(MDPDomain)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class D(RLDomain, FullyObservable)
        ```
    """
    pass


class POMDPDomain(Domain, SingleAgent, Sequential, EnumerableTransitions, Actions, UncertainInitialized, Markovian,
                  PartiallyObservable, Rewards):
    """This is a typical Partially Observable Markov Decision Process domain class.

    This helper class can be used as an alternate base class for domains, inheriting the following:

    - Domain
    - SingleAgent
    - Sequential
    - EnumerableTransitions
    - Actions
    - UncertainInitialized
    - Markovian
    - PartiallyObservable
    - Rewards

    Typical use:
    ```python
    class D(POMDPDomain)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class D(RLDomain, FullyObservable)
        ```
    """
    pass


class GoalMDPDomain(Domain, SingleAgent, Sequential, EnumerableTransitions, Actions, Goals, DeterministicInitialized,
                    Markovian, FullyObservable, PositiveCosts):
    """This is a typical Goal Markov Decision Process domain class.

    This helper class can be used as an alternate base class for domains, inheriting the following:

    - Domain
    - SingleAgent
    - Sequential
    - EnumerableTransitions
    - Actions
    - Goals
    - DeterministicInitialized
    - Markovian
    - FullyObservable
    - PositiveCosts

    Typical use:
    ```python
    class D(GoalMDPDomain)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class D(RLDomain, FullyObservable)
        ```
    """
    pass


class GoalPOMDPDomain(Domain, SingleAgent, Sequential, EnumerableTransitions, Actions, Goals, UncertainInitialized,
                      Markovian, PartiallyObservable, PositiveCosts):
    """This is a typical Goal Partially Observable Markov Decision Process domain class.

    This helper class can be used as an alternate base class for domains, inheriting the following:

    - Domain
    - SingleAgent
    - Sequential
    - EnumerableTransitions
    - Actions
    - Goals
    - UncertainInitialized
    - Markovian
    - PartiallyObservable
    - PositiveCosts

    Typical use:
    ```python
    class D(GoalPOMDPDomain)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class D(RLDomain, FullyObservable)
        ```
    """
    pass


class DeterministicPlanningDomain(Domain, SingleAgent, Sequential, DeterministicTransitions, Actions, Goals,
                                  DeterministicInitialized, Markovian, FullyObservable, PositiveCosts):
    """This is a typical deterministic planning domain class.

    This helper class can be used as an alternate base class for domains, inheriting the following:

    - Domain
    - SingleAgent
    - Sequential
    - DeterministicTransitions
    - Actions
    - Goals
    - DeterministicInitialized
    - Markovian
    - FullyObservable
    - PositiveCosts

    Typical use:
    ```python
    class D(DeterministicPlanningDomain)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class D(RLDomain, FullyObservable)
        ```
    """
    pass


_parallel_domain_ = None
_domain_factory_ = None

def parallel_domain_launcher(id):
    global _parallel_domain_, _domain_factory_
    _parallel_domain_.launch_domain_server(_domain_factory_, id)

class ExtendedManager(SyncManager):
    pass

ExtendedManager.register('LifoQueue', LifoQueue)

class ParallelDomain:
    """This class can be used to create and launch n domains in separate processes.
    Each domain listens for incoming domain requests.
    Each request can indicate which domain should serve it, otherwise the first available
    domain  is chosen and its id is returned to the incoming request.
    """
    def __init__(self, domain_factory, nb_domains = os.cpu_count()):
        global _parallel_domain_, _domain_factory_
        _parallel_domain_ = self
        _domain_factory_ = domain_factory
        self._manager = ExtendedManager()
        self._manager.start()
        self._waiting_jobs = [self._manager.Queue() for i in range(nb_domains)]
        self._sleeping_domains = self._manager.LifoQueue()
        self._active_domains = self._manager.list([False for i in range(nb_domains)])
        self._job_results = self._manager.list([None for i in range(nb_domains)])
        self._pool = Pool()
        for i in range(os.cpu_count()):
            self._pool.apply_async(parallel_domain_launcher, [i])
    
    def __del__(self):
        for i in range(len(self._job_results)):
            self._waiting_jobs[i].put(None)
        self._pool.close()
        self._pool.join()
    
    def nb_domains(self):
        return len(self._job_results)
    
    def launch_domain_server(self, domain_factory, id):
        domain = domain_factory()
        while True:
            self._active_domains[id] = False
            self._sleeping_domains.put(id)
            job = self._waiting_jobs[id].get()
            self._active_domains[id] = True
            self._job_results[id] = None
            if job is None:
                break
            else:
                self._job_results[id] = getattr(domain, job[0])(*job[1])
    
    def wake_up_domain(self, id=None):
        # in case of previous call to wake_up_domain
        # with a forced id, the elements in queue
        # self._sleeping_domains with that id could not
        # be popped out thus must be checked for actual inactivity
        if id is None:
            while True:
                tid = self._sleeping_domains.get()
                if not self._active_domains[tid]:
                    return tid
        else:
            return id
    
    def reset(self, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("reset", []))
        return mid
    
    def get_initial_state_distribution(self, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("get_initial_state_distribution", []))
        return mid
    
    def get_initial_state(self, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("get_initial_state", []))
        return mid
    
    def get_observation_space(self, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("get_observation_space", []))
        return mid
    
    def is_observation(self, observation, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("is_observation", [observation]))
        return mid
    
    def get_observation_distribution(self, state, action, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("get_observation_distribution", [state, action]))
        return mid
    
    def get_observation(self, state, action, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("get_observation", [state, action]))
        return mid
    
    def get_enabled_events(self, memory, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("get_enabled_events", [memory]))
        return mid
    
    def is_enabled_event(self, event, memory, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("is_enabled_event", [event, memory]))
        return mid
    
    def get_action_space(self, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("get_action_space", []))
        return mid

    def is_action(self, event, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("is_action", [event]))
        return mid
    
    def get_applicable_actions(self, memory, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("get_applicable_actions", [memory]))
        return mid
    
    def is_applicable_action(self, action, memory, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("is_applicable_action", [action, memory]))
        return mid
    
    def step(self, action, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("step", [action]))
        return mid
    
    def sample(self, memory, action, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("sample", [memory, action]))
        return mid
    
    def get_next_state_distribution(self, memory, action, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("get_next_state_distribution", [memory, action]))
        return mid
    
    def get_next_state(self, memory, action, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("get_next_state", [memory, action]))
        return mid
    
    def get_transition_value(self, memory, action, next_state, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("get_transition_value", [memory, action, next_state]))
        return mid
    
    def is_transition_value_dependent_on_next_state(self, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("is_transition_value_dependent_on_next_state", []))
        return mid
    
    def get_goals(self, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("get_goals", []))
        return mid
    
    def is_goal(self, observation, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("is_goal", [observation]))
        return mid
    
    def is_terminal(self, state, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("is_terminal", [state]))
        return mid
    
    def check_value(self, value, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("check_value", [value]))
        return mid
    
    def render(self, memory, id=None):
        mid = self.wake_up_domain(id)
        self._waiting_jobs[mid].put(("render", [memory]))
        return mid
    
    def get_result(self, id):
        r = self._job_results[id]
        return r if r is not None else None
