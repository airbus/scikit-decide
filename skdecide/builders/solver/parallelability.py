# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from skdecide.parallel_domains import PipeParallelDomain, ShmParallelDomain

__all__ = ["ParallelSolver"]


class ParallelSolver:
    """A solver must inherit this class if it wants to call several cloned parallel domains in separate concurrent processes.
    The solver is meant to be called either within a 'with' context statement, or to be cleaned up using the close() method.
    """

    def __init__(
        self,
        parallel: bool = False,
        shared_memory_proxy=None,
    ):
        """Creates a parallelizable solver

        # Parameters
        parallel: True if the solver is run in parallel mode.
        shared_memory_proxy: Shared memory proxy to use if not None, otherwise run piped parallel domains.



        """
        self._parallel = parallel
        self._shared_memory_proxy = shared_memory_proxy
        self._domain = None
        self._lambdas = []  # to define in the inherited class!
        self._ipc_notify = False  # to define in the inherited class!

    def _initialize(self):
        """Launches the parallel domains.
        This method requires to have previously recorded the self._domain_factory,
        the set of lambda functions passed to the solver's constructor (e.g. heuristic lambda for heuristic-based solvers),
        and whether the parallel domain jobs should notify their status via the IPC protocol (required when interacting with
        other programming languages like C++)
        """
        if self._parallel:
            if self._shared_memory_proxy is None:
                self._domain = PipeParallelDomain(
                    self._domain_factory,
                    lambdas=self._lambdas,
                    ipc_notify=self._ipc_notify,
                )
            else:
                self._domain = ShmParallelDomain(
                    self._domain_factory,
                    self._shared_memory_proxy,
                    lambdas=self._lambdas,
                    ipc_notify=self._ipc_notify,
                )
            # Launch parallel domains before created the algorithm object
            # otherwise spawning new processes (the default on Windows)
            # will fail trying to pickle the C++ underlying algorithm
            self._domain._launch_processes()
        else:
            self._domain = self._domain_factory()

    def close(self):
        """Joins the parallel domains' processes.
        Not calling this method (or not using the 'with' context statement)
        results in the solver forever waiting for the domain processes to exit.
        """
        if self._domain is not None and self._parallel:
            self._domain.close()
            self._domain = None

    def _cleanup(self):
        self.close()

    def get_domain(self):
        """
        Returns the domain, optionally creating a parallel domain if not already created.
        """
        if self._domain is None:
            self._initialize()
        return self._domain

    def call_domain_method(self, name, *args):
        """Calls a parallel domain's method.
        This is the only way to get a domain method for a parallel domain.
        """
        if self._parallel:
            process_id = getattr(self._domain, name)(*args)
            return self._domain.get_result(process_id)
        else:
            return getattr(self._domain, name)(*args)
