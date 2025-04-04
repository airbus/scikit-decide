"""PDDL domain based on plado library.

This code is inspired by
https://github.com/massle/plado/blob/60958c34105c01ec43f0dae8247dae889272220a/examples/skdecide_domain.py

"""
from __future__ import annotations

import itertools
import logging
from collections.abc import Hashable, Iterable
from enum import Enum
from typing import Optional, Union

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
)
from skdecide.hub.space.gym import (
    BoxSpace,
    DiscreteSpace,
    ListSpace,
    MaskableMultiDiscreteSpace,
)

logger = logging.getLogger(__name__)

try:
    import plado
except ImportError:
    plado_available = False
    logger.warning(
        "You need to install plado library to use PladoPddlDomain or PladoPPddlDomain!"
    )
    from fractions import Fraction

    Float = Fraction
else:
    plado_available = True
    from plado.parser import parse_and_normalize
    from plado.semantics.applicable_actions_generator import ApplicableActionsGenerator
    from plado.semantics.goal_checker import GoalChecker
    from plado.semantics.successor_generator import SuccessorGenerator
    from plado.semantics.task import State as PladoState
    from plado.semantics.task import Task
    from plado.utils import Float

SkAtomsType = tuple[tuple[tuple[int, ...], ...], ...]
AtomsType = list[set[tuple[int, ...]]]
SkFluentsType = tuple[tuple[tuple[tuple[int, ...], Float], ...], ...]
FluentsType = list[dict[tuple[int, ...], Float]]
GymVectorType = npt.NDArray[Union[int, float]]
GymMultidiscreteType = npt.NDArray[int]


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
    T_state = Union[SkPladoState, GymVectorType]  # Type of states
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
    ):
        self.domain_path: str = domain_path
        self.problem_path: str = problem_path
        self.state_encoding = state_encoding
        self.action_encoding = action_encoding
        domain, problem = parse_and_normalize(domain_path, problem_path)
        self.task: Task = Task(domain, problem)
        self.check_goal: GoalChecker = GoalChecker(self.task)
        self.aops_gen: ApplicableActionsGenerator = ApplicableActionsGenerator(
            self.task
        )
        self.succ_gen: SuccessorGenerator = SuccessorGenerator(self.task)
        self.total_cost: int | None = None
        for i, f in enumerate(self.task.functions):
            if f.name == "total-cost":
                self.total_cost = i
                break
        self.cost_functions: set[int] = set(
            [self.total_cost] if self.total_cost is not None else []
        )
        self._map_transition_value: dict[
            tuple[Hashable, Hashable, Hashable], D.T_value
        ] = {}
        self._init_state_encoding()
        self._init_action_encoding()

    def _init_state_encoding(self):
        if self.state_encoding == StateEncoding.NATIVE:
            ...
        elif self.state_encoding == StateEncoding.GYM_VECTOR:
            self._init_state_encoding_vector()
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
        else:
            raise NotImplementedError()

    def _translate_state_to_plado(self, state: D.T_state) -> PladoState:
        if self.state_encoding == StateEncoding.NATIVE:
            return state.to_plado()
        elif self.state_encoding == StateEncoding.GYM_VECTOR:
            return self._vector2state(vector=state)
        else:
            raise NotImplementedError()

    def _transform_state_to_hashable(self, state: D.T_state) -> Hashable:
        if self.state_encoding == StateEncoding.NATIVE:
            return state
        elif self.state_encoding == StateEncoding.GYM_VECTOR:
            return tuple(state)
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
        self._fluents_template = self.task.initial_state.fluents
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
