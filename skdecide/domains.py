# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This module contains base classes for quickly building domains."""
from __future__ import annotations

import os
import logging
from typing import NewType, Optional, Callable

from pathos.helpers import mp
from multiprocessing import Manager, Lock, Pipe, Process, Condition
from multiprocessing.sharedctypes import Array
import tempfile
from pynng import Push0

from skdecide.core import autocast_all
from skdecide.builders.domain.agent import MultiAgent, SingleAgent
from skdecide.builders.domain.concurrency import Parallel, Sequential
from skdecide.builders.domain.dynamics import Environment, Simulation, EnumerableTransitions, DeterministicTransitions
from skdecide.builders.domain.events import Events, Actions
from skdecide.builders.domain.goals import Goals
from skdecide.builders.domain.initialization import Initializable, UncertainInitialized, DeterministicInitialized
from skdecide.builders.domain.memory import History, Markovian
from skdecide.builders.domain.observability import PartiallyObservable, TransformedObservable, FullyObservable
from skdecide.builders.domain.value import Rewards, PositiveCosts
if False:  # trick to avoid circular import & IDE error ("Unresolved reference 'Solver'")
    from skdecide.solvers import Solver

__all__ = ['Domain', 'RLDomain', 'MultiAgentRLDomain', 'StatelessSimulatorDomain', 'MDPDomain', 'POMDPDomain',
           'GoalMDPDomain', 'GoalPOMDPDomain', 'DeterministicPlanningDomain']

logger = logging.getLogger('skdecide.domains')

logger.setLevel(logging.INFO)

if not len(logger.handlers):
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)
    logger.propagate = False


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


class ParallelDomain:
    """Base class for creating and launching n domains in separate processes.
    Each domain listens for incoming domain requests.
    Each request can indicate which domain should serve it, otherwise the first available
    domain i is chosen and its id is returned to the incoming request.
    """
    def __init__(self, domain_factory, nb_domains = os.cpu_count()):
        self._domain_factory = domain_factory
        self._call_i = None
        self._call_domain = False
        self._call_result = None
        self._active_domains = Array('b', [True for i in range(nb_domains)], lock=True)
        self._temp_connections = [tempfile.NamedTemporaryFile() for i in range(nb_domains)]
        self._ipc_connections = ['ipc://' + f.name + '.ipc' for f in self._temp_connections]
        self._processes = [None] * nb_domains
        self._ongoing_session = False
    
    def start_session(self):
        raise NotImplementedError

    def end_session(self):
        raise NotImplementedError

    def launch(self, i, name, *args):
        raise NotImplementedError

    def wait_job(self, i):  # using process connections (see below)
        raise NotImplementedError

    def get_proc_connections(self):  # process connections for use in python
        raise NotImplementedError
    
    def get_ipc_connections(self):  # inter-process connections for use with C++
        return self._ipc_connections
    
    def get_parallel_capacity(self):
        return self.nb_domains()
    
    def nb_domains(self):
        return len(self._processes)
    
    def wake_up_domain(self, i=None):
        if i is None:
            while True:
                for j, v in enumerate(self._active_domains):
                    if not v:
                        return j
        else:
            return i
    
    def reset(self, i=None):
        return self.launch(i, 'reset')
    
    def get_initial_state_distribution(self, i=None):
        return self.launch(i, 'get_initial_state_distribution')
    
    def get_initial_state(self, i=None):
        return self.launch(i, 'get_initial_state')
    
    def get_observation_space(self, i=None):
        return self.launch(i, 'get_observation_space')
    
    def is_observation(self, observation, i=None):
        return self.launch(i, 'is_observation', observation)
    
    def get_observation_distribution(self, state, action, i=None):
        return self.launch(i, 'get_observation_distribution', state, action)
    
    def get_observation(self, state, action, i=None):
        return self.launch(i, 'get_observation', state, action)
    
    def get_enabled_events(self, memory, i=None):
        return self.launch(i, 'get_enabled_events', memory)
    
    def is_enabled_event(self, event, memory, i=None):
        return self.launch(i, 'is_enabled_event', event, memory)
    
    def get_action_space(self, i=None):
        return self.launch(i, 'get_action_space')

    def is_action(self, event, i=None):
        return self.launch(i, 'is_action', event)
    
    def get_applicable_actions(self, memory, i=None):
        return self.launch(i, 'get_applicable_actions', memory)
    
    def is_applicable_action(self, action, memory, i=None):
        return self.launch(i, 'is_applicable_action', action, memory)
    
    def step(self, action, i=None):
        return self.launch(i, 'step', action)
    
    def sample(self, memory, action, i=None):
        return self.launch(i, 'sample', memory, action)
    
    def get_next_state_distribution(self, memory, action, i=None):
        return self.launch(i, 'get_next_state_distribution', memory, action)
    
    def get_next_state(self, memory, action, i=None):
        return self.launch(i, 'get_next_state', memory, action)
    
    def get_transition_value(self, memory, action, next_state, i=None):
        return self.launch(i, 'get_transition_value', memory, action, next_state)
    
    def is_transition_value_dependent_on_next_state(self, i=None):
        return self.launch(i, 'is_transition_value_dependent_on_next_state')
    
    def get_goals(self, i=None):
        return self.launch(i, 'get_goals')
    
    def is_goal(self, observation, i=None):
        return self.launch(i, 'is_goal', observation)
    
    def is_terminal(self, state, i=None):
        return self.launch(i, 'is_terminal', state)
    
    def check_value(self, value, i=None):
        return self.launch(i, 'check_value', value)
    
    def render(self, memory, i=None):
        return self.launch(i, 'render', memory)
    
    def call(self, i, function, *args):
        self._call_i = i
        self._call_domain = False
        mi = function(self, *args)  # will most probably call __getattr__.method below
        self._call_i = None
        if not self._call_domain:  # function is a lambda not calling the original domain
            self._call_result = mi
            return -1
        else:
            return mi
    
    # The original sequential domain may have methods we don't know
    def __getattr__(self, name):
        def method(*args, i=self._call_i):
            self._call_domain = True
            return self.launch(i, name, *args)
        return method


def _launch_domain_server_(domain_factory, i, lock, active_domains,
                           job_results, conn, ipc_conn, logger):
    domain = domain_factory()
    if ipc_conn is not None:
        pusher = Push0()
        pusher.dial(ipc_conn)
    while True:
        active_domains[i] = False
        job = conn.recv()
        lock.acquire()
        active_domains[i] = True
        job_results[i] = None
        lock.release()
        if job is None:
            if ipc_conn is not None:
                pusher.close()
            break
        else:
            try:
                r = getattr(domain, job[0])(*job[1])
                lock.acquire()
                job_results[i] = r
                lock.release()
                if ipc_conn is not None:
                    pusher.send(b'')
                else:
                    conn.send('')
            except Exception as e:
                logger.error(rf'/!\ Unable to perform job {job[0]}: {e}')

class PipeParallelDomain(ParallelDomain):
    """This class can be used to create and launch n domains in separate processes.
    Each domain listens for incoming domain requests.
    Each request can indicate which domain should serve it, otherwise the first available
    domain i is chosen and its id is returned to the incoming request.
    """
    def __init__(self, domain_factory, nb_domains = os.cpu_count()):
        super().__init__(domain_factory, nb_domains)
        self._manager = Manager()
        self._waiting_jobs = [None] * nb_domains
        self._job_results = self._manager.list([None for i in range(nb_domains)])
        self._lock = Lock()
        logger.info(rf'Using {nb_domains} parallel piped domains')
    
    def get_proc_connections(self):
        return self._waiting_jobs
    
    def wait_job(self, i):
        self._waiting_jobs[i].recv()
    
    def start_session(self, ipc_notify=False):
        if not self._ongoing_session:
            self._ongoing_session = True
            for i in range(len(self._job_results)):
                pparent, pchild = Pipe()
                self._waiting_jobs[i] = pparent
                self._processes[i] = mp.Process(target=_launch_domain_server_,
                                                args=[self._domain_factory, i, self._lock, self._active_domains,
                                                      self._job_results, pchild, self._ipc_connections[i] if ipc_notify else None, logger])
                self._processes[i].start()
            # Waits for all jobs to be launched and waiting each for requests
            while True in set(self._active_domains):
                continue
    
    def end_session(self):
        if self._ongoing_session:
            self._ongoing_session = False
            for i in range(len(self._job_results)):
                self._waiting_jobs[i].send(None)
                self._processes[i].join()
                self._processes[i].close()
                self._processes[i] = None
    
    def launch(self, i, name, *args):
        try:
            mi = self.wake_up_domain(i)
            self._waiting_jobs[mi].send((name, args))
            return mi
        except Exception as e:
            logger.error(rf'/!\ Unable to launch job {name}: {e}')
    
    def get_result(self, i):
        if i >= 0:
            self._lock.acquire()
            r = self._job_results[i]
            self._job_results[i] = None
            self._lock.release()
            return r
        else:  # we called a lambda function without using the original domain => main thread execution
            return self._call_result


def _shm_launch_domain_server_(domain_factory, i, active_domains,
                               shm_proxy, shm_registers, shm_types, shm_sizes,
                               rsize, shm_arrays, shm_names, shm_params,
                               cond, ipc_conn, logger):
    domain = domain_factory()
    if ipc_conn is not None:
        pusher = Push0()
        pusher.dial(ipc_conn)
    
    def get_string(s):
        for i, c in enumerate(s):
            if c == b'\x00':
                return s[:i].decode()
        return s.decode()

    while True:
        active_domains[i] = False
        with cond:
            cond.wait()
        active_domains[i] = True
        if shm_names[i][0] == b'\x00':
            if ipc_conn is not None:
                pusher.close()
            break
        else:
            try:
                job_name = get_string(shm_names[i])
                job_args = []
                for p in shm_params[i]:
                    if p >= 0:
                        sz = shm_sizes[shm_types[p].__name__]
                        if sz > 1:
                            si = (i * rsize) + p
                            job_args.append(shm_proxy.decode(shm_types[p], shm_arrays[si:(si + sz)]))
                        else:
                            job_args.append(shm_proxy.decode(shm_types[p], shm_arrays[(i * rsize) + p]))
                    else:
                        break  # no more args
                result = getattr(domain, job_name)(*job_args)
                shm_params[i][:] = [-1] * len(shm_params[i])
                if type(result) is not tuple:
                    result = (result,)
                if result[0] is not None:
                    type_counters = {}
                    for j, r in enumerate(result):
                        res_name = type(r).__name__
                        (start, end) = shm_registers[res_name]
                        if res_name in type_counters:
                            type_counters[res_name] += 1
                            k = type_counters[res_name]
                            if (k >= end):
                                raise IndexError('''No more available register for type {}. 
                                                    Please increase the number of registers 
                                                    for that type.'''.format(res_name))
                        else:
                            type_counters[res_name] = start
                            k = start
                        shm_params[i][j] = k
                        sz = shm_sizes[res_name]
                        if sz > 1:
                            si = (i * rsize) + k
                            shm_proxy.encode(r, shm_arrays[si:(si + sz)])
                        else:
                            shm_proxy.encode(r, shm_arrays[(i * rsize) + k])
                if ipc_conn is not None:
                    pusher.send(b'')
                else:
                    with cond:
                        cond.notify_all()
            except Exception as e:
                logger.error(rf'/!\ Unable to perform job {job_name}: {e}')

class ShmParallelDomain(ParallelDomain):
    """This class can be used to create and launch n domains in separate processes
    with shared memory between the Python processes.
    Each domain listens for incoming domain requests.
    Each request can indicate which domain should serve it, otherwise the first available
    domain is chosen and its id is returned to the incoming request.
    """
    def __init__(self, domain_factory, shm_proxy, nb_domains = os.cpu_count()):
        super().__init__(domain_factory, nb_domains)
        self._conditions = [Condition() for i in range(nb_domains)]
        self._shm_proxy = shm_proxy
        self._shm_registers = {}  # Maps from registered method parameter types to vectorized array ranges
        self._shm_types = {}  # Maps from register index to type
        self._shm_sizes = {}  # Maps from register method parameter types to number of arrays encoding each type
        self._shm_arrays = []  # Methods' vectorized parameters
        self._rsize = 0  # Total size of the register (updated below)
        self._shm_names = [None] * nb_domains  # Vectorized methods' names
        self._shm_params = [None] * nb_domains  # Indices of methods' vectorized parameters
        for i in range(nb_domains):
            j = 0
            for r in shm_proxy.register():
                for k in range(r[1]):
                    m = shm_proxy.initialize(r[0])
                    if type(m) == list or type(m) == tuple:
                        if i == 0 and k == 0:  # do it once for all the domains and redundant initializers
                            self._shm_sizes[r[0].__name__] = len(m)
                            self._shm_registers[r[0].__name__] = (j, j + (r[1] * len(m)))
                            self._shm_types.update({kk: r[0] for kk in range(j, j + (r[1] * len(m)), len(m))})
                            self._rsize += (r[1] * len(m))
                        self._shm_arrays.extend(m)
                        j += len(m)
                    else:
                        if i == 0 and k == 0:  # do it once for all the domains and redundant initializers
                            self._shm_sizes[r[0].__name__] = 1
                            self._shm_registers[r[0].__name__] = (j, j + r[1])
                            self._shm_types.update({kk: r[0] for kk in range(j, j + r[1])})
                            self._rsize += r[1]
                        self._shm_arrays.append(m)
                        j += 1
            self._shm_names[i] = Array('c', bytearray(100))
            self._shm_params[i] = Array('i', [-1] * sum(r[1] for r in shm_proxy.register()))
        logger.info(rf'Using {nb_domains} parallel shared memory domains')
    
    def get_proc_connections(self):
        return self._conditions
    
    def wait_job(self, i):
        with self._conditions[i]:
            self._conditions[i].wait()
    
    def start_session(self, ipc_notify=False):
        if not self._ongoing_session:
            self._ongoing_session = True
            for i in range(len(self._processes)):
                self._processes[i] = mp.Process(target=_shm_launch_domain_server_,
                                                args=[self._domain_factory, i, self._active_domains,
                                                      self._shm_proxy.copy(),
                                                      dict(self._shm_registers),
                                                      dict(self._shm_types),
                                                      dict(self._shm_sizes),
                                                      self._rsize, list(self._shm_arrays), list(self._shm_names), list(self._shm_params),
                                                      self._conditions[i], self._ipc_connections[i] if ipc_notify else None, logger])
                self._processes[i].start()
            # Waits for all jobs to be launched and waiting each for requests
            while True in set(self._active_domains):
                continue
    
    def end_session(self):
        if self._ongoing_session:
            self._ongoing_session = False
            for i in range(len(self._processes)):
                self._shm_names[i][:] = bytearray(len(self._shm_names[i]))  # reset with null bytes
                self._shm_params[i][:] = [-1] * len(self._shm_params[i])
                with self._conditions[i]:
                    self._conditions[i].notify_all()
                self._processes[i].join()
                self._processes[i].close()
                self._processes[i] = None
    
    def launch(self, i, name, *args):
        try:
            mi = self.wake_up_domain(i)
            self._shm_names[mi][:] = bytearray(name, encoding='utf-8') + \
                                     bytearray(len(self._shm_names[mi]) - len(name))
            self._shm_params[mi][:] = [-1] * len(self._shm_params[mi])
            type_counters = {}
            for j, a in enumerate(args):
                arg_name = type(a).__name__
                (start, end) = self._shm_registers[arg_name]
                if arg_name in type_counters:
                    type_counters[arg_name] += self._shm_sizes[arg_name]
                    k = type_counters[arg_name]
                    if (k >= end):
                        raise IndexError('''No more available register for type {}. 
                                            Please increase the number of registers 
                                            for that type.'''.format(arg_name))
                else:
                    type_counters[arg_name] = start
                    k = start
                self._shm_params[mi][j] = k
                sz = self._shm_sizes[arg_name]
                if sz > 1:
                    si = (mi * self._rsize) + k
                    self._shm_proxy.encode(a, self._shm_arrays[si:(si + sz)])
                else:
                    self._shm_proxy.encode(a, self._shm_arrays[(mi * self._rsize) + k])
            with self._conditions[mi]:
                self._conditions[mi].notify_all()
            return mi
        except Exception as e:
            logger.error(rf'/!\ Unable to launch job {name}: {e}')
    
    def get_result(self, i):
        results = []
        if i >= 0:
            for r in self._shm_params[i]:
                if r >= 0:
                    sz = self._shm_sizes[self._shm_types[r].__name__]
                    if sz > 1:
                        si = (i * self._rsize) + r
                        results.append(self._shm_proxy.decode(self._shm_types[r], self._shm_arrays[si:(si + sz)]))
                    else:
                        results.append(self._shm_proxy.decode(self._shm_types[r], self._shm_arrays[(i * self._rsize) + r]))
                else:
                    break  # no more params
            return results if len(results) > 1 else results[0] if len(results) > 0 else None
        else:  # we called a lambda function without using the original domain => main thread execution
            return self._call_result
