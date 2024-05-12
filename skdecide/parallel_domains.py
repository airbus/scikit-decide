# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This module contains helper classes to manage domains for parallel compurations."""

from __future__ import annotations

import logging
import os
import tempfile

# Following import is required to make Enum objects serializable
# (useful when multiprocessing and pickling domains that use Enum classes)
import dill
from pathos.helpers import mp
from pynng import Push0

logger = logging.getLogger(__name__)

dill.settings["byref"] = True


class ParallelDomain:
    """Base class for creating and launching n domains in separate processes.
    Each domain listens for incoming domain requests.
    Each request can indicate which domain should serve it, otherwise the first available
    domain i is chosen and its id is returned to the incoming request.
    """

    def __init__(
        self, domain_factory, lambdas=None, nb_domains=os.cpu_count(), ipc_notify=False
    ):
        self._domain_factory = domain_factory
        self._lambdas = lambdas
        self._active_domains = mp.Array(
            "b", [False for i in range(nb_domains)], lock=True
        )
        self._initializations = [
            mp.Value("b", False, lock=True) for i in range(nb_domains)
        ]
        self._conditions = [mp.Condition() for i in range(nb_domains)]
        self._temp_connections = [None] * nb_domains
        self._ipc_connections = [None] * nb_domains
        self._processes = [None] * nb_domains
        self._ipc_notify = ipc_notify

    def open_ipc_connection(self, i):
        self._temp_connections[i] = tempfile.NamedTemporaryFile(delete=True)
        self._ipc_connections[i] = "ipc://" + self._temp_connections[i].name + ".ipc"

    def close_ipc_connection(self, i):
        self._temp_connections[i].close()
        self._ipc_connections[i] = None

    def _launch_processes(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def __enter__(self):
        self._launch_processes()
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def launch(self, i, name, *args):
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
                        self._active_domains[j] = True
                        return j
        else:
            self._active_domains[i] = True
            return i

    def reset(self, i=None):
        return self.launch(i, "reset")

    def get_initial_state_distribution(self, i=None):
        return self.launch(i, "get_initial_state_distribution")

    def get_initial_state(self, i=None):
        return self.launch(i, "get_initial_state")

    def get_observation_space(self, i=None):
        return self.launch(i, "get_observation_space")

    def is_observation(self, observation, i=None):
        return self.launch(i, "is_observation", observation)

    def get_observation_distribution(self, state, action, i=None):
        return self.launch(i, "get_observation_distribution", state, action)

    def get_observation(self, state, action, i=None):
        return self.launch(i, "get_observation", state, action)

    def get_enabled_events(self, memory, i=None):
        return self.launch(i, "get_enabled_events", memory)

    def is_enabled_event(self, event, memory, i=None):
        return self.launch(i, "is_enabled_event", event, memory)

    def get_action_space(self, i=None):
        return self.launch(i, "get_action_space")

    def is_action(self, event, i=None):
        return self.launch(i, "is_action", event)

    def get_applicable_actions(self, memory, i=None):
        return self.launch(i, "get_applicable_actions", memory)

    def is_applicable_action(self, action, memory, i=None):
        return self.launch(i, "is_applicable_action", action, memory)

    def step(self, action, i=None):
        return self.launch(i, "step", action)

    def sample(self, memory, action, i=None):
        return self.launch(i, "sample", memory, action)

    def get_next_state_distribution(self, memory, action, i=None):
        return self.launch(i, "get_next_state_distribution", memory, action)

    def get_next_state(self, memory, action, i=None):
        return self.launch(i, "get_next_state", memory, action)

    def get_transition_value(self, memory, action, next_state, i=None):
        return self.launch(i, "get_transition_value", memory, action, next_state)

    def is_transition_value_dependent_on_next_state(self, i=None):
        return self.launch(i, "is_transition_value_dependent_on_next_state")

    def get_goals(self, i=None):
        return self.launch(i, "get_goals")

    def is_goal(self, observation, i=None):
        return self.launch(i, "is_goal", observation)

    def is_terminal(self, state, i=None):
        return self.launch(i, "is_terminal", state)

    def check_value(self, value, i=None):
        return self.launch(i, "check_value", value)

    def render(self, memory, i=None):
        return self.launch(i, "render", memory)

    # Call a lambda function (usually involves the original domain)
    def call(self, i, lambda_id, *args):
        return self.launch(i, lambda_id, *args)

    # The original sequential domain may have methods we don't know
    def __getattr__(self, name):
        def method(*args, i=None):
            return self.launch(i, name, *args)

        return method

    # Bypass __getattr_.method() when serializing the class.
    # Required on Windows when spawning the main process.
    def __getstate__(self):
        d = self.__dict__.copy()
        del d["_temp_connections"]  # we cannot serialize a file
        return d

    # Bypass __getattr_.method() when serializing the class.
    # Required on Windows when spawning the main process.
    def __setstate__(self, state):
        self.__dict__ = state
        # TODO: reopen the temp connection from _ipc_connections


def _launch_domain_server_(
    domain_factory, lambdas, i, job_results, conn, init, cond, ipc_conn, logger
):
    domain = domain_factory()

    if ipc_conn is not None:
        pusher = Push0()
        pusher.dial(
            ipc_conn, block=False
        )  # WARNING: recent pynng updates only work with non-blocking dials!

    with cond:
        init.value = True
        cond.notify_all()  # inform the parent process that we are ready to process requests

    while True:
        job = conn.recv()
        job_results[i] = None
        if job is None:
            if ipc_conn is not None:
                pusher.close()
            conn.close()
            break
        else:
            try:
                if isinstance(job[0], str):  # job[0] is a domain class' method
                    r = getattr(domain, job[0])(*job[1])
                else:  # job[0] is a lambda function
                    r = lambdas[job[0]](domain, *job[1])
                job_results[i] = r
                if ipc_conn is not None:
                    pusher.send(b"0")  # send success
                conn.send("0")  # send success
            except Exception as e:
                logger.error(rf"/!\ Unable to perform job {job[0]}: {e}")
                if ipc_conn is not None:
                    pusher.send(str(e).encode(encoding="UTF-8"))  # send error message
                else:
                    conn.send(str(e))  # send failure (!= 0)


class PipeParallelDomain(ParallelDomain):
    """This class can be used to create and launch n domains in separate processes.
    Each domain listens for incoming domain requests.
    Each request can indicate which domain should serve it, otherwise the first available
    domain i is chosen and its id is returned to the incoming request.
    """

    def __init__(
        self, domain_factory, lambdas=None, nb_domains=os.cpu_count(), ipc_notify=False
    ):
        super().__init__(domain_factory, lambdas, nb_domains, ipc_notify)
        self._manager = mp.Manager()
        self._waiting_jobs = [None] * nb_domains
        self._job_results = self._manager.list([None for i in range(nb_domains)])
        logger.info(rf"Using {nb_domains} parallel piped domains")

    def get_proc_connections(self):
        return self._waiting_jobs

    def launch(self, i, function, *args):
        if not any(self._processes):
            self._launch_processes()
        try:
            mi = self.wake_up_domain(i)
            self._waiting_jobs[mi].send((function, args))
            return mi
        except Exception as e:
            if isinstance(function, str):
                logger.error(rf"/!\ Unable to launch job {function}: {e}")
            else:
                logger.error(rf"/!\ Unable to launch job lambdas[{function}]: {e}")

    def get_result(self, i):
        self._waiting_jobs[i].recv()
        r = self._job_results[i]
        self._job_results[i] = None
        self._active_domains[i] = False
        return r

    def _launch_processes(self):
        for i in range(len(self._job_results)):
            self.open_ipc_connection(i)
            pparent, pchild = mp.Pipe()
            self._waiting_jobs[i] = pparent
            self._processes[i] = mp.Process(
                target=_launch_domain_server_,
                args=[
                    self._domain_factory,
                    self._lambdas,
                    i,
                    self._job_results,
                    pchild,
                    self._initializations[i],
                    self._conditions[i],
                    self._ipc_connections[i] if self._ipc_notify else None,
                    logger,
                ],
            )
            self._processes[i].start()
        # Waits for all jobs to be launched and waiting each for requests
        for i in range(len(self._job_results)):
            with self._conditions[i]:
                self._conditions[i].wait_for(
                    lambda: bool(self._initializations[i].value) == True
                )

    def close(self):
        for i in range(len(self._job_results)):
            self._initializations[i].value = False
            self._waiting_jobs[i].send(None)
            self._processes[i].join()
            self._processes[i].close()
            self._waiting_jobs[i].close()
            self._processes[i] = None
            self.close_ipc_connection(i)


def _shm_launch_domain_server_(
    domain_factory,
    lambdas,
    i,
    shm_proxy,
    shm_registers,
    shm_types,
    shm_sizes,
    rsize,
    shm_arrays,
    shm_lambdas,
    shm_names,
    shm_params,
    init,
    activation,
    done,
    cond,
    ipc_conn,
    logger,
):
    domain = domain_factory()

    if ipc_conn is not None:
        pusher = Push0()
        pusher.dial(
            ipc_conn, block=False
        )  # WARNING: recent pynng updates only work with non-blocking dials!

    with cond:
        init.value = True
        cond.notify_all()  # inform the parent process that we are ready to process requests

    def get_string(s):
        for i, c in enumerate(s):
            if c == b"\x00":
                return s[:i].decode()
        return s.decode()

    while True:
        with cond:
            cond.wait_for(lambda: bool(activation.value) == True)
            activation.value = False
        if int(shm_lambdas[i].value) == -1 and shm_names[i][0] == b"\x00":
            if ipc_conn is not None:
                pusher.close()
            break
        else:
            try:
                job_args = []
                for p in shm_params[i]:
                    if p >= 0:
                        sz = shm_sizes[shm_types[p].__name__]
                        if sz > 1:
                            si = (i * rsize) + p
                            job_args.append(
                                shm_proxy.decode(
                                    shm_types[p], shm_arrays[si : (si + sz)]
                                )
                            )
                        else:
                            job_args.append(
                                shm_proxy.decode(
                                    shm_types[p], shm_arrays[(i * rsize) + p]
                                )
                            )
                    else:
                        break  # no more args
                if (
                    int(shm_lambdas[i].value) == -1
                ):  # we are working with a domain class' method
                    result = getattr(domain, get_string(shm_names[i]))(*job_args)
                else:  # we are working with a lambda function
                    result = lambdas[int(shm_lambdas[i].value)](domain, *job_args)
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
                            if k >= end:
                                raise IndexError(
                                    """No more available register for type {}.
                                                    Please increase the number of registers
                                                    for that type.""".format(
                                        res_name
                                    )
                                )
                        else:
                            type_counters[res_name] = start
                            k = start
                        shm_params[i][j] = k
                        sz = shm_sizes[res_name]
                        if sz > 1:
                            si = (i * rsize) + k
                            shm_proxy.encode(r, shm_arrays[si : (si + sz)])
                        else:
                            shm_proxy.encode(r, shm_arrays[(i * rsize) + k])
                if ipc_conn is not None:
                    pusher.send(b"0")  # send success
            except Exception as e:
                if int(shm_lambdas[i].value) == -1:
                    logger.error(
                        rf"/!\ Unable to perform job {get_string(shm_names[i])}: {e}"
                    )
                else:
                    logger.error(
                        rf"/!\ Unable to perform job {int(shm_lambdas[i].value)}: {e}"
                    )
                if ipc_conn is not None:
                    pusher.send(str(e).encode(encoding="UTF-8"))  # send error message
            with cond:
                done.value = True
                cond.notify_all()  # send finished status (no success nor failure information)


class ShmParallelDomain(ParallelDomain):
    """This class can be used to create and launch n domains in separate processes
    with shared memory between the Python processes.
    Each domain listens for incoming domain requests.
    Each request can indicate which domain should serve it, otherwise the first available
    domain is chosen and its id is returned to the incoming request.
    """

    def __init__(
        self,
        domain_factory,
        shm_proxy,
        lambdas=None,
        nb_domains=os.cpu_count(),
        ipc_notify=False,
    ):
        super().__init__(domain_factory, lambdas, nb_domains, ipc_notify)
        self._activations = [mp.Value("b", False, lock=True) for i in range(nb_domains)]
        self._dones = [mp.Value("b", False, lock=True) for i in range(nb_domains)]
        self._shm_proxy = shm_proxy
        self._shm_registers = (
            {}
        )  # Maps from registered method parameter types to vectorized array ranges
        self._shm_types = {}  # Maps from register index to type
        self._shm_sizes = (
            {}
        )  # Maps from register method parameter types to number of arrays encoding each type
        self._shm_arrays = []  # Methods' vectorized parameters
        self._rsize = 0  # Total size of the register (updated below)
        self._shm_lambdas = [None] * nb_domains  # Vectorized lambdas' ids
        self._shm_names = [None] * nb_domains  # Vectorized methods' names
        self._shm_params = [
            None
        ] * nb_domains  # Indices of methods' vectorized parameters
        for i in range(nb_domains):
            j = 0
            for r in shm_proxy.register():
                for k in range(r[1]):
                    m = shm_proxy.initialize(r[0])
                    if type(m) == list or type(m) == tuple:
                        if (
                            i == 0 and k == 0
                        ):  # do it once for all the domains and redundant initializers
                            self._shm_sizes[r[0].__name__] = len(m)
                            self._shm_registers[r[0].__name__] = (
                                j,
                                j + (r[1] * len(m)),
                            )
                            self._shm_types.update(
                                {
                                    kk: r[0]
                                    for kk in range(j, j + (r[1] * len(m)), len(m))
                                }
                            )
                            self._rsize += r[1] * len(m)
                        self._shm_arrays.extend(m)
                        j += len(m)
                    else:
                        if (
                            i == 0 and k == 0
                        ):  # do it once for all the domains and redundant initializers
                            self._shm_sizes[r[0].__name__] = 1
                            self._shm_registers[r[0].__name__] = (j, j + r[1])
                            self._shm_types.update(
                                {kk: r[0] for kk in range(j, j + r[1])}
                            )
                            self._rsize += r[1]
                        self._shm_arrays.append(m)
                        j += 1
            self._shm_lambdas[i] = mp.Value("i", -1, lock=True)
            self._shm_names[i] = mp.Array("c", bytearray(100))
            self._shm_params[i] = mp.Array(
                "i", [-1] * sum(r[1] for r in shm_proxy.register())
            )
        logger.info(rf"Using {nb_domains} parallel shared memory domains")

    def get_proc_connections(self):
        return (self._activations, self._conditions)

    def launch(self, i, function, *args):
        if not any(self._processes):
            self._launch_processes()
        try:
            mi = self.wake_up_domain(i)
            if isinstance(function, str):  # function is a domain class' method
                self._shm_lambdas[mi].value = -1
                self._shm_names[mi][:] = bytearray(
                    function, encoding="utf-8"
                ) + bytearray(len(self._shm_names[mi]) - len(function))
            else:  # function is a lambda id
                self._shm_lambdas[mi].value = int(function)
                self._shm_names[mi][:] = bytearray(
                    len(self._shm_names[mi])
                )  # reset with null bytes
            self._shm_params[mi][:] = [-1] * len(self._shm_params[mi])
            type_counters = {}
            for j, a in enumerate(args):
                arg_name = type(a).__name__
                (start, end) = self._shm_registers[arg_name]
                if arg_name in type_counters:
                    type_counters[arg_name] += self._shm_sizes[arg_name]
                    k = type_counters[arg_name]
                    if k >= end:
                        raise IndexError(
                            """No more available register for type {}.
                                            Please increase the number of registers
                                            for that type.""".format(
                                arg_name
                            )
                        )
                else:
                    type_counters[arg_name] = start
                    k = start
                self._shm_params[mi][j] = k
                sz = self._shm_sizes[arg_name]
                if sz > 1:
                    si = (mi * self._rsize) + k
                    self._shm_proxy.encode(a, self._shm_arrays[si : (si + sz)])
                else:
                    self._shm_proxy.encode(a, self._shm_arrays[(mi * self._rsize) + k])
            with self._conditions[mi]:
                self._activations[mi].value = True
                self._conditions[mi].notify_all()
            return mi
        except Exception as e:
            if isinstance(function, str):
                logger.error(rf"/!\ Unable to launch job {function}: {e}")
            else:
                logger.error(rf"/!\ Unable to launch job lambdas[{function}]: {e}")

    def get_result(self, i):
        with self._conditions[i]:
            self._conditions[i].wait_for(lambda: bool(self._dones[i].value) == True)
            self._dones[i].value = False
        results = []
        for r in self._shm_params[i]:
            if r >= 0:
                sz = self._shm_sizes[self._shm_types[r].__name__]
                if sz > 1:
                    si = (i * self._rsize) + r
                    results.append(
                        self._shm_proxy.decode(
                            self._shm_types[r], self._shm_arrays[si : (si + sz)]
                        )
                    )
                else:
                    results.append(
                        self._shm_proxy.decode(
                            self._shm_types[r], self._shm_arrays[(i * self._rsize) + r]
                        )
                    )
            else:
                break  # no more params
        self._active_domains[i] = False
        return results if len(results) > 1 else results[0] if len(results) > 0 else None

    def _launch_processes(self):
        for i in range(len(self._processes)):
            self.open_ipc_connection(i)
            self._processes[i] = mp.Process(
                target=_shm_launch_domain_server_,
                args=[
                    self._domain_factory,
                    self._lambdas,
                    i,
                    self._shm_proxy.copy(),
                    dict(self._shm_registers),
                    dict(self._shm_types),
                    dict(self._shm_sizes),
                    self._rsize,
                    list(self._shm_arrays),
                    list(self._shm_lambdas),
                    list(self._shm_names),
                    list(self._shm_params),
                    self._initializations[i],
                    self._activations[i],
                    self._dones[i],
                    self._conditions[i],
                    self._ipc_connections[i] if self._ipc_notify else None,
                    logger,
                ],
            )
            self._processes[i].start()
        # Waits for all jobs to be launched and waiting each for requests
        for i in range(len(self._processes)):
            with self._conditions[i]:
                self._conditions[i].wait_for(
                    lambda: bool(self._initializations[i].value) == True
                )

    def close(self):
        for i in range(len(self._processes)):
            self._initializations[i].value = False
            self._shm_lambdas[i].value = -1
            self._shm_names[i][:] = bytearray(
                len(self._shm_names[i])
            )  # reset with null bytes
            self._shm_params[i][:] = [-1] * len(self._shm_params[i])
            with self._conditions[i]:
                self._activations[i].value = True
                self._conditions[i].notify_all()
            self._processes[i].join()
            self._processes[i].close()
            self._processes[i] = None
            self.close_ipc_connection(i)
