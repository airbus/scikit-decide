from collections.abc import Callable
from typing import Any, Optional

from pyRDDLGym_gurobi.core.planner import (
    GurobiOnlineController,
    GurobiPlan,
    GurobiStraightLinePlan,
)
from pyRDDLGym_jax.core.planner import (
    JaxBackpropPlanner,
    JaxOfflineController,
    load_config,
)

from skdecide import Solver
from skdecide.builders.solver import Policies
from skdecide.hub.domain.rddl import RDDLDomain


class D(RDDLDomain):
    pass


class RDDLJaxSolver(Solver, Policies):
    T_domain = D

    def __init__(
        self, domain_factory: Callable[[], RDDLDomain], config: Optional[str] = None
    ):
        Solver.__init__(self, domain_factory=domain_factory)
        self._domain = domain_factory()
        if config is not None:
            self.planner_args, _, self.train_args = load_config(config)

    @classmethod
    def _check_domain_additional(cls, domain: D) -> bool:
        return hasattr(domain, "rddl_gym_env")

    def _solve(self, from_memory: Optional[D.T_state] = None) -> None:
        planner = JaxBackpropPlanner(
            rddl=self._domain.rddl_gym_env.model,
            **(self.planner_args if self.planner_args is not None else {}),
        )
        self.controller = JaxOfflineController(
            planner, **(self.train_args if self.train_args is not None else {})
        )

    def _sample_action(self, observation: D.T_observation) -> D.T_event:
        return self.controller.sample_action(observation)

    def _is_policy_defined_for(self, observation: D.T_observation) -> bool:
        return True


class RDDLGurobiSolver(Solver, Policies):
    T_domain = D

    def __init__(
        self,
        domain_factory: Callable[[], RDDLDomain],
        plan: Optional[GurobiPlan] = None,
        rollout_horizon=5,
        model_params: Optional[dict[str, Any]] = None,
    ):
        Solver.__init__(self, domain_factory=domain_factory)
        self._domain = domain_factory()
        self.rollout_horizon = rollout_horizon
        if plan is None:
            self.plan = GurobiStraightLinePlan()
        else:
            self.plan = plan
        if model_params is None:
            self.model_params = {"NonConvex": 2, "OutputFlag": 0}
        else:
            self.model_params = model_params

    @classmethod
    def _check_domain_additional(cls, domain: D) -> bool:
        return hasattr(domain, "rddl_gym_env")

    def _solve(self, from_memory: Optional[D.T_state] = None) -> None:
        self.controller = GurobiOnlineController(
            rddl=self._domain.rddl_gym_env.model,
            plan=self.plan,
            rollout_horizon=self.rollout_horizon,
            model_params=self.model_params,
        )

    def _sample_action(self, observation: D.T_observation) -> D.T_event:
        return self.controller.sample_action(observation)

    def _is_policy_defined_for(self, observation: D.T_observation) -> bool:
        return True
