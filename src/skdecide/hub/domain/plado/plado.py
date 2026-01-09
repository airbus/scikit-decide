"""PDDL domain based on plado library.

This code is inspired by
https://github.com/massle/plado/blob/60958c34105c01ec43f0dae8247dae889272220a/examples/skdecide_domain.py

"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Hashable, Iterable
from enum import Enum
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from skdecide import (
    DiscreteDistribution,
    Domain,
    ImplicitSpace,
    Space,
    TransitionOutcome,
    Value,
)
from skdecide.builders.domain import (
    Actions,
    DeterministicInitialized,
    DeterministicTransitions,
    EnumerableTransitions,
    FullyObservable,
    Goals,
    Markovian,
    PositiveCosts,
    Sequential,
    SingleAgent,
    TransformedObservable,
)
from skdecide.hub.domain.plado.llg_encoder import LLGEncoder, NodeLabel
from skdecide.hub.space.gym import (
    BoxSpace,
    DiscreteSpace,
    GymSpace,
    ListSpace,
    MaskableMultiDiscreteSpace,
)

logger = logging.getLogger(__name__)

try:
    from plado.parser import parse_and_normalize
    from plado.semantics.applicable_actions_generator import ApplicableActionsGenerator
    from plado.semantics.goal_checker import GoalChecker
    from plado.semantics.successor_generator import SuccessorGenerator
    from plado.semantics.task import State as PladoState
    from plado.semantics.task import Task
    from plado.utils import Float
except ImportError:
    plado_available = False
    logger.warning(
        "You need to install plado library to use PladoPddlDomain or PladoPPddlDomain!"
    )
    from fractions import Fraction

    Float = Fraction
else:
    plado_available = True


SkAtomsType = tuple[tuple[tuple[int, ...], ...], ...]
AtomsType = list[set[tuple[int, ...]]]
SkFluentsType = tuple[tuple[tuple[tuple[int, ...], Float], ...], ...]
FluentsType = list[dict[tuple[int, ...], Float]]
GymVectorType = npt.NDArray[Union[int, float]]
GymMultidiscreteType = npt.NDArray[int]
GraphObjectEdgeFeatureType = npt.NDArray[Union[int, float]]
GraphObjectNodeFeatureType = npt.NDArray[Union[int, float]]


class SkPladoState:
    """Wrapper around plado.State to ensure hashability and comparability.

    The fluents corresponding to the cost functions are also automatically set to 0.

    """

    def __init__(
        self,
        atoms: SkAtomsType,
        fluents: SkFluentsType,
        cost_functions: Optional[set[int]] = None,
    ):
        # replace total_cost fluent value by 0
        if cost_functions is not None and len(cost_functions) > 0:
            fluents = tuple(
                ((tuple(), Float(0)),) if f in cost_functions else fluents_
                for f, fluents_ in enumerate(fluents)
            )
        self.atoms = atoms
        self.fluents = fluents

    @staticmethod
    def from_plado(
        state: PladoState, cost_functions: Optional[set[int]] = None
    ) -> SkPladoState:
        atoms: SkAtomsType = tuple(
            tuple(sorted(predicate_atoms)) for predicate_atoms in state.atoms
        )
        fluents: SkFluentsType = tuple(
            tuple(sorted(fluents_.items())) for f, fluents_ in enumerate(state.fluents)
        )
        return SkPladoState(atoms=atoms, fluents=fluents, cost_functions=cost_functions)

    def to_plado(self) -> PladoState:
        state = PladoState(0, 0)
        state.atoms = [set(predicate_atoms) for predicate_atoms in self.atoms]
        state.fluents = [dict(fluents_) for fluents_ in self.fluents]
        return state

    def __hash__(self):
        return hash((self.atoms, self.fluents))

    def __eq__(self, o) -> bool:
        return (
            isinstance(o, SkPladoState)
            and self.atoms == o.atoms
            and self.fluents == o.fluents
        )


PladoAction = tuple[int, tuple[int, ...]]


class D(
    Domain,
    SingleAgent,
    Sequential,
    EnumerableTransitions,
    Actions,
    Goals,
    DeterministicInitialized,
    Markovian,
    FullyObservable,
    PositiveCosts,
):
    T_state = Union[
        SkPladoState, GymVectorType, gym.spaces.GraphInstance
    ]  # Type of states
    T_observation = T_state  # Type of observations
    T_event = Union[PladoAction, int, GymMultidiscreteType]  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of test results
    T_info = None  # Type of additional information in environment outcome


class ObservationSpace(Space[D.T_observation]):
    def __init__(
        self,
        predicate_arities: Iterable[int],
        function_arities: Iterable[int],
        num_objects: int,
    ):
        self.predicate_arities: tuple[int, ...] = tuple(predicate_arities)
        self.function_arities: tuple[int, ...] = tuple(function_arities)
        self.num_objects: int = num_objects

    def contains(self, x: D.T_observation) -> bool:
        if len(x.atoms) != len(self.predicate_arities) or len(x.fluents) != len(
            self.function_arities
        ):
            return False
        for predicate_atoms, predicate_arities in zip(x.atoms, self.predicate_arities):
            for params in predicate_atoms:
                if len(params) != predicate_arities or (
                    predicate_arities > 0
                    and (min(params) < 0 or max(params) >= self.num_objects)
                ):
                    return False
        for function_fluents, function_arities in zip(x.fluents, self.function_arities):
            for params, _ in function_fluents:
                if len(params) != function_arities or (
                    function_arities > 0
                    and (min(params) < 0 or max(params) >= self.num_objects)
                ):
                    return False
        return True


class ActionSpace(Space[D.T_event]):
    def __init__(self, action_arities: Iterable[int], num_objects: int):
        self.action_arities: tuple[int, ...] = tuple(action_arities)
        self.num_objects: int = num_objects

    def contains(self, a: D.T_event) -> bool:
        return (
            a[0] >= 0
            and a[0] < len(self.action_arities)
            and len(a[1]) == self.action_arities[a[0]]
            and min(a[1]) >= 0
            and max(a[1]) < self.num_objects
        )


class StateEncoding(Enum):
    NATIVE = "native"  # SkPladoState: hashable version of plado.State
    GYM_VECTOR = "gym-vector"  # flat numpy.array (in particular for RL algorithms)
    GYM_GRAPH_LLG = "gym-graph-llg"
    """Lifted learning graph encoding actions and predicates schemes + instance specificities.

    Corresponds more or less to the "LLG" from
    Chen, D. Z., Thiébaux, S., & Trevizan, F. (2024).
    Learning Domain-Independent Heuristics for Grounded and Lifted Planning.
    Proceedings of the AAAI Conference on Artificial Intelligence, 38(18), 20078-20086.
    https://doi.org/10.1609/aaai.v38i18.29986

    """


class ObservationEncoding(Enum):
    GYM_GRAPH_OBJECTS = "gym-graph-objects"
    """Encode state as graph whose nodes are the objects and edges are predicates true in common.

    Corresponds to the "Object Binary Structure" from
    Horčík, R., & Šír, G. (2024). Expressiveness of Graph Neural Networks in Planning Domains.
    Proceedings of the International Conference on Automated Planning and Scheduling, 34(1), 281-289.
    https://ojs.aaai.org/index.php/ICAPS/article/view/31486

    """


class ActionEncoding(Enum):
    NATIVE = "native"  # tuple (i_action, (i_param_0, ..., i_param_p)) returned by ApplicableActionsGenerator
    GYM_DISCRETE = "gym-discrete"  # indice among all actions possible (tractable only for very small problems)
    GYM_MULTIDISCRETE = "gym-multidiscrete"  # np.array([i_action, i_param_0, ..., i_param_p]) very similar to "native", but "gym compatible


class BasePladoDomain(D):
    """Base class for scikit-decide domains based on plado library."""

    def __init__(
        self,
        domain_path: str,
        problem_path: str,
        state_encoding: StateEncoding = StateEncoding.NATIVE,
        action_encoding: ActionEncoding = ActionEncoding.NATIVE,
        llg_encoder_kwargs: Optional[dict[str, Any]] = None,
    ):
        """

        Args:
            domain_path:
            problem_path:
            state_encoding:
            action_encoding:
            llg_encoder_kwargs:

        """
        self.domain_path: str = domain_path
        self.problem_path: str = problem_path
        self.state_encoding = state_encoding
        self.action_encoding = action_encoding
        if llg_encoder_kwargs is None:
            self.llg_encoder_kwargs = {}
        else:
            self.llg_encoder_kwargs = llg_encoder_kwargs
        domain, problem = parse_and_normalize(domain_path, problem_path)
        self.task: Task = Task(domain, problem)
        self.check_goal: GoalChecker = GoalChecker(self.task)
        self.aops_gen: ApplicableActionsGenerator = ApplicableActionsGenerator(
            self.task
        )
        self.succ_gen: SuccessorGenerator = SuccessorGenerator(self.task)
        self.total_cost: Optional[int] = None
        for i, f in enumerate(self.task.functions):
            if f.name == "total-cost":
                self.total_cost = i
                break
        if self.total_cost is None:
            self.cost_functions = set()
        else:
            self.cost_functions = {self.total_cost}
        self._map_transition_value: dict[
            tuple[Hashable, Hashable, Hashable], D.T_value
        ] = {}
        self._init_state_encoding()
        self._init_action_encoding()

    def get_action_components_node_flag_indices(self) -> list[Optional[int]]:
        """Give the indices of the node features encoding which node can be used for each action component.

        This is used by autoregressive GNN based solvers that will predict
        a node of a different type for each component of the action. The node type is encoded in a node feature,
        potentially different for each component (action nodes then object nodes).

        Not implemented if action_encoding is not multidiscrete;

        Returns:
            list of node feature indices corresponding to each action component.

        """
        if self.action_encoding == ActionEncoding.GYM_MULTIDISCRETE:
            if self.state_encoding == StateEncoding.GYM_GRAPH_LLG:
                if self._llg_encoder.encode_actions:
                    return [self._llg_encoder.map_nodelabel2int[NodeLabel.ACTION]] + [
                        self._llg_encoder.map_nodelabel2int[NodeLabel.OBJECT]
                    ] * self._max_action_arity
                else:
                    return [None] + [
                        self._llg_encoder.map_nodelabel2int[NodeLabel.OBJECT]
                    ] * self._max_action_arity
            else:
                return (1 + self._max_action_arity) * [None]
        else:
            raise NotImplementedError()

    def _init_state_encoding(self):
        if self.state_encoding == StateEncoding.NATIVE:
            ...
        elif self.state_encoding == StateEncoding.GYM_VECTOR:
            self._init_state_encoding_vector()
        elif self.state_encoding == StateEncoding.GYM_GRAPH_LLG:
            self._llg_encoder = LLGEncoder(
                task=self.task,
                cost_functions=self.cost_functions,
                **self.llg_encoder_kwargs,
            )
        else:
            raise NotImplementedError()

    def _init_action_encoding(self):
        if self.action_encoding == ActionEncoding.NATIVE:
            ...
        elif self.action_encoding == ActionEncoding.GYM_DISCRETE:
            self._init_action_encoding_discrete()
        elif self.action_encoding == ActionEncoding.GYM_MULTIDISCRETE:
            self._init_action_encoding_multidiscrete()
        else:
            raise NotImplementedError()

    def _translate_state_from_plado(self, state: PladoState) -> D.T_state:
        if self.state_encoding == StateEncoding.NATIVE:
            return SkPladoState.from_plado(
                state=state, cost_functions=self.cost_functions
            )
        elif self.state_encoding == StateEncoding.GYM_VECTOR:
            return self._state2vector(state)
        elif self.state_encoding == StateEncoding.GYM_GRAPH_LLG:
            return self._llg_encoder.encode(state)
        else:
            raise NotImplementedError()

    def repr_obs_as_plado(self, obs: D.T_observation) -> str:
        """Return a string representation of the observation similar to plado representation."""
        plado_state = self._translate_state_to_plado(obs)
        return f"PladoState(atoms={plado_state.atoms}, fluents={plado_state.fluents})"

    def _translate_state_to_plado(self, state: D.T_state) -> PladoState:
        if self.state_encoding == StateEncoding.NATIVE:
            return state.to_plado()
        elif self.state_encoding == StateEncoding.GYM_VECTOR:
            return self._vector2state(vector=state)
        elif self.state_encoding == StateEncoding.GYM_GRAPH_LLG:
            return self._llg_encoder.decode(state)
        else:
            raise NotImplementedError()

    def _transform_state_to_hashable(self, state: D.T_state) -> Hashable:
        if self.state_encoding == StateEncoding.NATIVE:
            return state
        elif self.state_encoding == StateEncoding.GYM_VECTOR:
            return tuple(state)
        elif self.state_encoding == StateEncoding.GYM_GRAPH_LLG:
            return SkPladoState.from_plado(
                state=self._llg_encoder.decode(state),
                cost_functions=self.cost_functions,
            )
        else:
            raise NotImplementedError()

    def _transform_action_to_hashable(self, action: D.T_event) -> Hashable:
        if self.action_encoding == ActionEncoding.GYM_MULTIDISCRETE:
            return tuple(action)
        else:
            return action

    def _translate_action_from_plado(self, action: PladoAction) -> D.T_event:
        if self.action_encoding == ActionEncoding.NATIVE:
            return action
        elif self.action_encoding == ActionEncoding.GYM_DISCRETE:
            return self._map_action2idx[action]
        elif self.action_encoding == ActionEncoding.GYM_MULTIDISCRETE:
            return self._action2multidiscrete(action)
        else:
            raise NotImplementedError()

    def _translate_action_to_plado(self, action: D.T_event) -> PladoAction:
        if self.action_encoding == ActionEncoding.NATIVE:
            return action
        elif self.action_encoding == ActionEncoding.GYM_DISCRETE:
            return self._map_idx2action[action]
        elif self.action_encoding == ActionEncoding.GYM_MULTIDISCRETE:
            return self._multidiscrete2action(action)
        else:
            raise NotImplementedError()

    def _action2multidiscrete(self, action: PladoAction) -> GymMultidiscreteType:
        action_id, params_id = action
        return np.array(
            (action_id,)
            + params_id
            + (-1,) * (self._max_action_arity - len(params_id)),
            dtype=int,
        )

    def _multidiscrete2action(self, action: GymMultidiscreteType) -> PladoAction:
        action_id = int(action[0])
        action_arity = self.task.actions[action_id].parameters
        return action_id, tuple(int(i) for i in action[1 : action_arity + 1])

    def _get_cost_from_state(self, state: PladoState) -> int:
        if self.total_cost is None:
            return 1  # assume unit cost
        return int(state.fluents[self.total_cost][tuple()])

    def _get_initial_state_(self) -> D.T_state:
        return self._translate_state_from_plado(self.task.initial_state)

    def _get_observation_space_(self) -> Space[D.T_observation]:
        if self.state_encoding == StateEncoding.NATIVE:
            return ObservationSpace(
                (
                    len(self.task.predicates[p].parameters)
                    for p in range(self.task.num_fluent_predicates)
                ),
                (len(f.parameters) for f in self.task.functions),
                len(self.task.objects),
            )
        elif self.state_encoding == StateEncoding.GYM_VECTOR:
            if self._fluents_vector_size == 0:
                dtype = int
            else:
                dtype = float

            return BoxSpace(
                low=np.concatenate(
                    (
                        np.zeros((self._atoms_vector_size,)),
                        np.full((self._fluents_vector_size), -np.inf),
                    )
                ),
                high=np.concatenate(
                    (
                        np.ones((self._atoms_vector_size,)),
                        np.full((self._fluents_vector_size), +np.inf),
                    )
                ),
                dtype=dtype,
            )
        elif self.state_encoding == StateEncoding.GYM_GRAPH_LLG:
            return GymSpace(self._llg_encoder.graph_space)
        else:
            raise NotImplementedError()

    def _get_action_space_(self) -> Space[D.T_event]:
        if self.action_encoding == ActionEncoding.NATIVE:
            return ActionSpace(
                (a.parameters for a in self.task.actions), len(self.task.objects)
            )
        elif self.action_encoding == ActionEncoding.GYM_DISCRETE:
            return DiscreteSpace(n=len(self._map_idx2action))
        elif self.action_encoding == ActionEncoding.GYM_MULTIDISCRETE:
            n_objects = len(self.task.objects)
            n_actions = len(self.task.actions)
            return MaskableMultiDiscreteSpace(
                nvec=[n_actions] + self._max_action_arity * [n_objects]
            )
        else:
            raise NotImplementedError()

    def _get_goals_(self) -> Space[D.T_observation]:
        return ImplicitSpace(
            lambda s: self.check_goal(self._translate_state_to_plado(s))
        )

    def _is_terminal(self, state: D.T_state) -> D.T_predicate:
        return self._is_terminal_from_plado(self._translate_state_to_plado(state))

    def _is_terminal_from_plado(self, pladostate: PladoState) -> D.T_predicate:
        return self.check_goal(
            pladostate
        ) or not self._has_applicable_actions_from_plado(pladostate)

    def _has_applicable_actions_from_plado(self, pladostate: PladoState) -> bool:
        applicable_plado_actions_generator = self.aops_gen(pladostate)
        try:
            next(applicable_plado_actions_generator)
        except StopIteration:
            return False
        else:
            return True

    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        applicable_plado_actions_generator = self.aops_gen(
            self._translate_state_to_plado(memory)
        )
        return ListSpace(
            [
                self._translate_action_from_plado(plado_action)
                for plado_action in applicable_plado_actions_generator
            ]
        )

    def _state_sample(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> TransitionOutcome[
        D.T_state,
        D.T_agent[Value[D.T_value]],
        D.T_agent[D.T_predicate],
        D.T_agent[D.T_info],
    ]:
        """Compute one sample of the transition's dynamics.

        As plado compute together next state distribution and transition value,
        it is more efficient to override this method rather than using the default one calling separately
        `_get_next_state_distribution()` and `_get_transition_value()`.
        Avoids also multiple translation pladostate -> skstate -> pladostate ...

        """
        successors = self.succ_gen(
            self._translate_state_to_plado(memory),
            self._translate_action_to_plado(action),
        )
        pladostates_with_proba = [(succ, float(prob)) for succ, prob in successors]
        pladostate = DiscreteDistribution(pladostates_with_proba).sample()
        skstate = self._translate_state_from_plado(pladostate)
        value = Value(cost=self._get_cost_from_state(pladostate))
        termination = self._is_terminal_from_plado(pladostate)

        return TransitionOutcome(
            state=skstate, value=value, termination=termination, info=None
        )

    def _get_next_state_distribution(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> DiscreteDistribution[D.T_state]:
        (
            skstates_with_proba,
            map_transition_value,
        ) = self._get_transitions_proba_and_value(memory=memory, action=action)
        # store values to avoid calling again self.succ_gen in _get_transition_value()
        # override previously stored values
        self._map_transition_value = map_transition_value

        return DiscreteDistribution(skstates_with_proba)

    def _get_transitions_proba_and_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> tuple[
        list[tuple[D.T_state, float]],
        dict[tuple[Hashable, Hashable, Hashable], D.T_value],
    ]:
        successors = self.succ_gen(
            self._translate_state_to_plado(memory),
            self._translate_action_to_plado(action),
        )
        skstates_with_proba = [
            (self._translate_state_from_plado(succ), float(prob))
            for succ, prob in successors
        ]
        map_transition_value = {
            (
                self._transform_state_to_hashable(memory),
                self._transform_action_to_hashable(action),
                self._transform_state_to_hashable(skstate),
            ): self._get_cost_from_state(pladostate)
            for (pladostate, _), (skstate, _) in zip(successors, skstates_with_proba)
        }

        return skstates_with_proba, map_transition_value

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        key = (
            self._transform_state_to_hashable(memory),
            self._transform_action_to_hashable(action),
            self._transform_state_to_hashable(next_state),
        )
        try:
            cost = self._map_transition_value[key]
        except KeyError:
            # not called just after self._get_next_state_distribution_? or concurrent calls?
            _, map_transition_value = self._get_transitions_proba_and_value(
                memory=memory, action=action
            )
            try:
                cost = map_transition_value[key]
            except KeyError:
                raise ValueError(
                    f"The transition (memory={memory}, action={action}, next_state={next_state}) "
                    "is not an existing transition!"
                )

        return Value(cost=cost)

    def _init_state_encoding_vector(self):
        self._fluents_template: FluentsType = self.task.initial_state.fluents
        self._fluents_vector_size = len(self._fluents2vector(self._fluents_template))

        n_objects = len(self.task.objects)
        self._map_idx_to_atom_by_predicate = []
        self._map_atom_to_idx_by_predicate = []
        for predicate in self.task.predicates[: self.task.num_fluent_predicates]:
            map_idx_to_atom = []
            map_atom_to_idx = {}
            for i_atom, atom in enumerate(
                itertools.product(*(range(n_objects) for param in predicate.parameters))
            ):
                map_idx_to_atom.append(atom)
                map_atom_to_idx[atom] = i_atom
            self._map_idx_to_atom_by_predicate.append(map_idx_to_atom)
            self._map_atom_to_idx_by_predicate.append(map_atom_to_idx)

        atoms_vector_size_by_predicate = tuple(
            int(np.prod(tuple(n_objects for param in predicate.parameters)))
            # should be n_objects_by_type[param.type_name] but normalization => all same object types
            for predicate in self.task.predicates[: self.task.num_fluent_predicates]
        )
        self._start_atoms_vector_by_predicate = np.cumsum(
            (0,) + atoms_vector_size_by_predicate[:-1]
        )
        self._end_atoms_vector_by_predicate = np.cumsum(atoms_vector_size_by_predicate)
        self._atoms_vector_size = self._end_atoms_vector_by_predicate[-1]

    def _fluents2vector(self, fluents: FluentsType) -> npt.NDArray[float]:
        return np.array(
            [
                float(fluent_value)
                for i_type, fluents_by_type in enumerate(fluents)
                if i_type not in self.cost_functions
                for fluent_params, fluent_value in fluents_by_type.items()
            ]
        )

    def _vector2fluents(self, vector: npt.NDArray[float]) -> FluentsType:
        vector_iter = iter(vector)
        return [
            {
                fluent_params: Float(next(vector_iter))
                if i_type not in self.cost_functions
                else Float(0)
                for fluent_params, fluent_value in fluents_by_type.items()
            }
            for i_type, fluents_by_type in enumerate(self._fluents_template)
        ]

    def _atoms2vector(self, atoms: AtomsType) -> npt.NDArray[int]:
        vector_atoms = np.zeros((self._atoms_vector_size,), dtype=int)
        for atoms_predicate, map_atom_to_idx, idx_shift in zip(
            atoms,
            self._map_atom_to_idx_by_predicate,
            self._start_atoms_vector_by_predicate,
        ):
            for atom in atoms_predicate:
                vector_atoms[map_atom_to_idx[atom] + idx_shift] = 1
        return vector_atoms

    def _vector2atoms(self, vector: npt.NDArray[int]) -> AtomsType:
        return list(
            set(map_idx_to_atom[idx] for idx in vector[start:end].nonzero()[0])
            for start, end, map_idx_to_atom in zip(
                self._start_atoms_vector_by_predicate,
                self._end_atoms_vector_by_predicate,
                self._map_idx_to_atom_by_predicate,
            )
        )

    def _state2vector(self, state: PladoState) -> npt.NDArray[Union[int, float]]:
        atoms_vector = self._atoms2vector(state.atoms)
        fluents_vector = self._fluents2vector(state.fluents)
        if len(fluents_vector) == 0:
            return atoms_vector  # keep dtype=int
        else:
            return np.concatenate((atoms_vector, fluents_vector))

    def _vector2state(self, vector: npt.NDArray[Union[int, float]]) -> PladoState:
        atoms_vector = vector[: self._atoms_vector_size]
        fluents_vector = vector[self._atoms_vector_size :]
        atoms = self._vector2atoms(atoms_vector)
        fluents = self._vector2fluents(fluents_vector)
        state = PladoState(0, 0)
        state.atoms = atoms
        state.fluents = fluents
        return state

    def _init_action_encoding_discrete(self):
        n_objects = len(self.task.objects)
        self._map_idx2action: list[PladoAction] = [
            (i_action_type, parameters)
            for i_action_type, action in enumerate(self.task.actions)
            for parameters in itertools.product(
                *(range(n_objects) for param in range(action.parameters))
            )
        ]

        self._map_action2idx: dict[PladoAction, int] = {
            action: i_action for i_action, action in enumerate(self._map_idx2action)
        }

    def _init_action_encoding_multidiscrete(self):
        self._max_action_arity = max(a.parameters for a in self.task.actions)


class D(
    Domain,
    SingleAgent,
    Sequential,
    EnumerableTransitions,
    Actions,
    Goals,
    DeterministicInitialized,
    Markovian,
    TransformedObservable,
    PositiveCosts,
):
    T_state = SkPladoState  # Type of states
    T_observation = Union[T_state, gym.spaces.GraphInstance]  # Type of observations
    T_event = Union[PladoAction, int, GymMultidiscreteType]  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of test results
    T_info = None  # Type of additional information in environment outcome


class BasePladoTransformedObservableDomain(D):
    """Base class for scikit-decide domains based on plado library, which are only partially observable.

    Some graph representation of the state are not injective and so lead to a `TransformedObservable` domain.

    # Attributes

    obs_encoding: observation (graph) encoding used
        - ObservationEncoding.GYM_GRAPH_OBJECTS: graph whose nodes are the objects and edges are
        predicates true in common.
        Corresponds to the "Object Binary Structure" from
        Horčík, R., & Šír, G. (2024). Expressiveness of Graph Neural Networks in Planning Domains.
        Proceedings of the International Conference on Automated Planning and Scheduling, 34(1), 281-289.
        https://ojs.aaai.org/index.php/ICAPS/article/view/31486

    """

    plado_domain_class: type[BasePladoDomain]

    def __init__(
        self,
        domain_path: str,
        problem_path: str,
        obs_encoding: ObservationEncoding = ObservationEncoding.GYM_GRAPH_OBJECTS,
        action_encoding: ActionEncoding = ActionEncoding.NATIVE,
        graph_objects_encode_static_facts: bool = True,
    ):
        self.obs_encoding = obs_encoding
        self.graph_objects_encode_static_facts = graph_objects_encode_static_facts
        self.plado_domain = self.plado_domain_class(
            domain_path=domain_path,
            problem_path=problem_path,
            state_encoding=StateEncoding.NATIVE,
            action_encoding=action_encoding,
        )
        self._init_obs_encoding()

    @property
    def action_encoding(self) -> ActionEncoding:
        return self.plado_domain.action_encoding

    @property
    def domain_path(self) -> str:
        return self.plado_domain.domain_path

    @property
    def problem_path(self) -> str:
        return self.plado_domain.problem_path

    @property
    def task(self) -> Task:
        return self.plado_domain.task

    def _init_obs_encoding(self):
        if self.obs_encoding == ObservationEncoding.GYM_GRAPH_OBJECTS:
            self._init_obs_encoding_graph_objects()
        else:
            raise NotImplementedError()

    def _state_sample(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> TransitionOutcome[
        D.T_state,
        D.T_agent[Value[D.T_value]],
        D.T_agent[D.T_predicate],
        D.T_agent[D.T_info],
    ]:
        return self.plado_domain._state_sample(memory, action)

    def _get_next_state_distribution(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> DiscreteDistribution[D.T_state]:
        return self.plado_domain._get_next_state_distribution(
            memory=memory, action=action
        )

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        return self.plado_domain._get_transition_value(
            memory=memory, action=action, next_state=next_state
        )

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        return self.plado_domain._is_terminal(state=state)

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return self.plado_domain._get_action_space_()

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        return self.plado_domain._get_applicable_actions_from(memory=memory)

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return ImplicitSpace(lambda o: self._is_goal(o))

    def _is_goal(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_predicate]:
        """Check whether the observation is a goal

        Warning:
        Here we implement it based on current state, because the obs is not sufficient to assert it.
        So the result will be correct only if the observation is actually the current observation.

        """
        return self.plado_domain.check_goal(
            self.plado_domain._translate_state_to_plado(self._memory)
        )

    def _get_initial_state_(self) -> D.T_state:
        return self.plado_domain._get_initial_state_()

    def _get_observation(
        self,
        state: D.T_state,
        action: Optional[D.T_agent[D.T_concurrency[D.T_event]]] = None,
    ) -> D.T_agent[D.T_observation]:
        if self.obs_encoding == ObservationEncoding.GYM_GRAPH_OBJECTS:
            return self._state2graphobjects(state)
        else:
            raise NotImplementedError()

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        if self.obs_encoding == ObservationEncoding.GYM_GRAPH_OBJECTS:
            node_low = np.array(
                [0]
                * (
                    len(self.i_global_fluent_predicates)
                    + len(self.i_global_static_predicates)
                )
                + [-np.inf] * len(self.i_global_fluents)
                + [0]
                * (
                    len(self.i_node_fluent_predicates)
                    + len(self.i_node_static_predicates)
                )
                + [-np.inf] * len(self.i_node_fluents),
                dtype=self.dtype_node_features,
            )
            node_high = np.array(
                [1]
                * (
                    len(self.i_global_fluent_predicates)
                    + len(self.i_global_static_predicates)
                )
                + [np.inf] * len(self.i_global_fluents)
                + [1]
                * (
                    len(self.i_node_fluent_predicates)
                    + len(self.i_node_static_predicates)
                )
                + [np.inf] * len(self.i_node_fluents),
                dtype=self.dtype_node_features,
            )
            edge_low = np.array(
                [0]
                * (
                    len(self.i_edge_fluent_predicates)
                    + len(self.i_edge_static_predicates)
                )
                + [-np.inf] * len(self.i_edge_fluents),
                dtype=self.dtype_edge_features,
            )
            edge_high = np.array(
                [1]
                * (
                    len(self.i_edge_fluent_predicates)
                    + len(self.i_edge_static_predicates)
                )
                + [np.inf] * len(self.i_edge_fluents),
                dtype=self.dtype_edge_features,
            )
            return GymSpace(
                gym.spaces.Graph(
                    node_space=gym.spaces.Box(
                        low=node_low, high=node_high, dtype=self.dtype_node_features
                    ),
                    edge_space=gym.spaces.Box(
                        low=edge_low, high=edge_high, dtype=self.dtype_edge_features
                    ),
                )
            )
        else:
            raise NotImplementedError()

    @property
    def task(self) -> Task:
        return self.plado_domain.task

    @property
    def cost_functions(self) -> set[int]:
        return self.plado_domain.cost_functions

    def _init_obs_encoding_graph_objects(self):
        # Split predicates and fluents according to global, node and edge features, according to their arity.

        # global features
        # NB: global features will be encoded as node features, as gymnasium.spaces.Graph does not allow global features
        # and preset GNN from pytorch_geometric do not take them into account
        self.i_global_fluent_predicates = [  # predicates changing according to actions
            i_type
            for i_type, p in enumerate(
                self.task.predicates[: self.task.num_fluent_predicates]
            )
            if len(p.parameters) == 0
        ]
        if self.graph_objects_encode_static_facts:
            self.i_global_static_predicates = [  # static predicates
                i_type
                for i_type, p in enumerate(
                    self.task.predicates[-self.task.num_static_predicates :]
                )
                if len(p.parameters) == 0
            ]
        else:
            self.i_global_static_predicates = []
        self.i_global_fluents = [  # numeric fluents
            i_type
            for i_type, f in enumerate(self.task.functions)
            if len(f.parameters) == 0 and i_type not in self.cost_functions
        ]
        # node features
        self.i_node_fluent_predicates = [  # predicates changing according to actions
            i_type
            for i_type, p in enumerate(
                self.task.predicates[: self.task.num_fluent_predicates]
            )
            if len(p.parameters) == 1
        ]
        if self.graph_objects_encode_static_facts:
            self.i_node_static_predicates = [  # static predicates
                i_type
                for i_type, p in enumerate(
                    self.task.predicates[-self.task.num_static_predicates :]
                )
                if len(p.parameters) == 1
            ]
        else:
            self.i_node_static_predicates = []
        self.i_node_fluents = [  # numeric fluents
            i_type
            for i_type, f in enumerate(self.task.functions)
            if len(f.parameters) == 1
            and i_type not in self.cost_functions  # remove total-cost fluent
        ]
        # edge features
        self.i_edge_fluent_predicates = [  # predicates changing according to actions
            i_type
            for i_type, p in enumerate(
                self.task.predicates[: self.task.num_fluent_predicates]
            )
            if len(p.parameters) > 1
        ]
        if self.graph_objects_encode_static_facts:
            self.i_edge_static_predicates = [  # static predicates
                i_type
                for i_type, p in enumerate(
                    self.task.predicates[-self.task.num_static_predicates :]
                )
                if (
                    len(p.parameters) > 1
                    and len(self.task.predicates)
                    - self.task.num_static_predicates
                    + i_type
                    != self.task.eq_predicate  # remove equality static predicate
                )
            ]
        else:
            self.i_edge_static_predicates = []
        self.i_edge_fluents = [  # numeric fluents
            i_type
            for i_type, f in enumerate(self.task.functions)
            if len(f.parameters) > 1 and i_type not in self.cost_functions
        ]

        n_global_features = (
            len(self.i_global_fluent_predicates)
            + len(self.i_global_static_predicates)
            + len(self.i_global_fluents)
        )
        self.n_node_features = (
            n_global_features
            + len(self.i_node_fluent_predicates)
            + len(self.i_node_static_predicates)
            + len(self.i_node_fluents)
        )
        self.n_edge_features = (
            len(self.i_edge_fluent_predicates)
            + len(self.i_edge_static_predicates)
            + len(self.i_edge_fluents)
        )

        self.dtype_edge_features = (
            np.int8 if len(self.i_edge_fluents) == 0 else np.float64
        )
        self.dtype_node_features = (
            np.int8 if len(self.i_node_fluents) == 0 else np.float64
        )

        self.n_nodes = len(self.task.objects)

    def _state2graphobjects(self, state: SkPladoState) -> gym.spaces.GraphInstance:
        nodes = np.zeros(
            (self.n_nodes, self.n_node_features), dtype=self.dtype_node_features
        )
        i_col = 0
        # encode global features (as node features)
        for i_type in self.i_global_fluent_predicates:
            if len(state.atoms[i_type]) > 0:  # 0-ary predicate is True
                nodes[:, i_col] = 1
            i_col += 1
        for i_type in self.i_global_static_predicates:
            if len(self.task.static_facts[i_type]) > 0:  # 0-ary predicate is True
                nodes[:, i_col] = 1
            i_col += 1
        for i_type in self.i_global_fluents:
            for params, value in state.fluents[i_type]:
                # fluents list not empty (but params empty)
                value = float(value)
                nodes[:, i_col] = value
            i_col += 1
        # encode node features
        for i_type in self.i_node_fluent_predicates:
            i_rows = [params[0] for params in state.atoms[i_type]]
            nodes[i_rows, i_col] = 1
            i_col += 1
        for i_type in self.i_node_static_predicates:
            i_rows = [params[0] for params in self.task.static_facts[i_type]]
            nodes[i_rows, i_col] = 1
            i_col += 1
        for i_type in self.i_node_fluents:
            for params, value in state.fluents[i_type]:
                value = float(value)
                i_row = params[0]  # only 1 param
                nodes[i_row, i_col] = value
            i_col += 1

        # encode edge_links and edge features
        map_edge2ind: dict[tuple[int, int], int] = {}
        edge_links: list[tuple[int, int]] = []
        edges = []
        i_col = 0

        for i_type in self.i_edge_fluent_predicates:
            for params in state.atoms[i_type]:
                self._update_edge_features_from_edge_predicate(
                    edges=edges,
                    edge_links=edge_links,
                    map_edge2ind=map_edge2ind,
                    i_col=i_col,
                    params=params,
                    value=1,
                )
            i_col += 1
        for i_type in self.i_edge_static_predicates:
            for params in self.task.static_facts[i_type]:
                self._update_edge_features_from_edge_predicate(
                    edges=edges,
                    edge_links=edge_links,
                    map_edge2ind=map_edge2ind,
                    i_col=i_col,
                    params=params,
                    value=1,
                )
            i_col += 1
        for i_type in self.i_edge_fluents:
            for params, value in state.fluents[i_type]:
                value = float(value)
                self._update_edge_features_from_edge_predicate(
                    edges=edges,
                    edge_links=edge_links,
                    map_edge2ind=map_edge2ind,
                    i_col=i_col,
                    params=params,
                    value=value,
                )
            i_col += 1

        return gym.spaces.GraphInstance(
            nodes=nodes,
            edge_links=np.array(edge_links, dtype=np.int_),
            edges=np.array(edges, dtype=self.dtype_edge_features),
        )

    def _update_edge_features_from_edge_predicate(
        self,
        edges: list[GraphObjectEdgeFeatureType],
        edge_links: list[tuple[int, int]],
        map_edge2ind: dict[tuple[int, int], int],
        i_col: int,
        params: tuple[int, ...],
        value: Union[int, float] = 1,
    ) -> None:
        # loop on couples made from params (potentially len(params) > 2)
        # here we get an undirected graph, a predicate between 3 objects
        # leads to all possible edges between the 3 objects
        for edge in itertools.permutations(params, 2):
            try:
                i_edge = map_edge2ind[edge]
            except KeyError:
                i_edge = len(edges)
                edges.append(np.zeros((self.n_edge_features,)))
                edge_links.append(edge)
                map_edge2ind[edge] = i_edge
            edges[i_edge][i_col] = value


class PladoPddlDomain(DeterministicTransitions, BasePladoDomain):
    """Wrapper around Plado Task for (deterministic) PDDL."""

    def _state_sample(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> TransitionOutcome[
        D.T_state,
        D.T_agent[Value[D.T_value]],
        D.T_agent[D.T_predicate],
        D.T_agent[D.T_info],
    ]:
        """Compute one sample of the transition's dynamics.

        As plado compute together next state distribution and transition value,
        it is more efficient to override this method rather than implementing separately
        `_get_next_state()` and `_get_transition_value()`.

        """
        successors = self.succ_gen(
            self._translate_state_to_plado(memory),
            self._translate_action_to_plado(action),
        )
        # deterministic => actual successor is the first in the list
        pladostate = successors[0][0]
        skstate = self._translate_state_from_plado(pladostate)
        value = Value(cost=self._get_cost_from_state(pladostate))
        termination = self._is_terminal_from_plado(pladostate)

        return TransitionOutcome(
            state=skstate, value=value, termination=termination, info=None
        )

    def _get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_state:
        (
            skstates_with_proba,
            map_transition_value,
        ) = self._get_transitions_proba_and_value(memory=memory, action=action)
        # store values to avoid calling again self.succ_gen in _get_transition_value()
        # override previously stored values
        self._map_transition_value = map_transition_value

        return skstates_with_proba[0][0]


class PladoPPddlDomain(BasePladoDomain):
    """Wrapper around Plado Task for Probabilistic PDDL."""

    ...


class PladoTransformedObservablePddlDomain(
    DeterministicTransitions, BasePladoTransformedObservableDomain
):
    """Wrapper around Plado Task for (deterministic) PDDL with observations transformed from state."""

    plado_domain_class = PladoPddlDomain


class PladoTransformedObservablePPddlDomain(BasePladoTransformedObservableDomain):
    """Wrapper around Plado Task for Probabilistic PDDL with observations transformed from state."""

    plado_domain_class = PladoPPddlDomain
