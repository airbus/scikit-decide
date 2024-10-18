import os
import shutil
from datetime import datetime
from typing import Any

import numpy as np
import pyRDDLGym
from gymnasium.spaces.utils import flatten, flatten_space
from pyRDDLGym import RDDLEnv
from pyRDDLGym.core.simulator import RDDLSimulator
from pyRDDLGym.core.visualizer.chart import ChartVisualizer
from pyRDDLGym.core.visualizer.movie import MovieGenerator
from pyRDDLGym.core.visualizer.viz import BaseViz
from pyRDDLGym_rl.core.env import SimplifiedActionRDDLEnv

from skdecide.builders.domain import FullyObservable, Renderable, UnrestrictedActions
from skdecide.core import Space, TransitionOutcome, Value
from skdecide.domains import RLDomain
from skdecide.hub.space.gym import GymSpace

try:
    import IPython
except ImportError:
    ipython_available = False
else:
    ipython_available = True
    from IPython.display import clear_output, display


class D(RLDomain, UnrestrictedActions, FullyObservable, Renderable):
    T_state = dict[str, Any]  # Type of states
    T_observation = T_state  # Type of observations
    T_event = np.array  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = None  # Type of additional information in environment outcome


class RDDLDomain(D):
    def __init__(
        self,
        rddl_domain: str,
        rddl_instance: str,
        base_class: type[RDDLEnv] = RDDLEnv,
        backend: type[RDDLSimulator] = RDDLSimulator,
        display_with_pygame: bool = True,
        display_within_jupyter: bool = False,
        visualizer: BaseViz = ChartVisualizer,
        movie_name: str = None,
        movie_dir: str = "rddl_movies",
        max_frames=1000,
        enforce_action_constraints=True,
        **kwargs
    ):
        self.rddl_gym_env = pyRDDLGym.make(
            rddl_domain,
            rddl_instance,
            base_class=base_class,
            backend=backend,
            enforce_action_constraints=enforce_action_constraints,
            **kwargs
        )
        self.display_within_jupyter = display_within_jupyter
        self.display_with_pygame = display_with_pygame
        self.movie_name = movie_name
        self._nb_step = 0
        if movie_name is not None:
            self.movie_path = os.path.join(movie_dir, movie_name)
            if not os.path.exists(self.movie_path):
                os.makedirs(self.movie_path)
            tmp_pngs = os.path.join(self.movie_path, "tmp_pngs")
            if os.path.exists(tmp_pngs):
                shutil.rmtree(tmp_pngs)
            os.makedirs(tmp_pngs)
            self.movie_gen = MovieGenerator(tmp_pngs, movie_name, max_frames=max_frames)
            self.rddl_gym_env.set_visualizer(visualizer, self.movie_gen)
        else:
            self.movie_gen = None
            self.rddl_gym_env.set_visualizer(visualizer)

    def _state_step(
        self, action: D.T_event
    ) -> TransitionOutcome[D.T_state, Value[D.T_value], D.T_predicate, D.T_info]:
        next_state, reward, terminated, truncated, _ = self.rddl_gym_env.step(action)
        termination = terminated or truncated
        if self.movie_gen is not None and (
            termination or self._nb_step >= self.movie_gen.max_frames - 1
        ):
            self.movie_gen.save_animation(self.movie_name)
            tmp_pngs = os.path.join(self.movie_path, "tmp_pngs")
            shutil.move(
                os.path.join(tmp_pngs, self.movie_name + ".gif"),
                os.path.join(
                    self.movie_path,
                    self.movie_name
                    + "_"
                    + str(datetime.now().strftime("%Y%m%d-%H%M%S"))
                    + ".gif",
                ),
            )
        self._nb_step += 1
        # TransitionOutcome and Value are scikit-decide types
        return TransitionOutcome(
            state=next_state, value=Value(reward=reward), termination=termination
        )

    def _get_action_space_(self) -> Space[D.T_event]:
        # Cast to skdecide's GymSpace
        return GymSpace(self.rddl_gym_env.action_space)

    def _state_reset(self) -> D.T_state:
        self._nb_step = 0
        # SkDecide only needs the state, not the info
        return self.rddl_gym_env.reset()[0]

    def _get_observation_space_(self) -> Space[D.T_observation]:
        # Cast to skdecide's GymSpace
        return GymSpace(self.rddl_gym_env.observation_space)

    def _render_from(self, memory: D.T_state = None, **kwargs: Any) -> Any:
        # We do not want the image to be displayed in a pygame window, but rather in this notebook
        rddl_gym_img = self.rddl_gym_env.render(to_display=self.display_with_pygame)
        if self.display_within_jupyter and ipython_available:
            clear_output(wait=True)
            display(rddl_gym_img)
        return rddl_gym_img


class RDDLDomainRL(RDDLDomain):
    def __init__(
        self,
        rddl_domain: str,
        rddl_instance: str,
        base_class: type[RDDLEnv] = SimplifiedActionRDDLEnv,
        backend: type[RDDLSimulator] = RDDLSimulator,
        display_with_pygame: bool = True,
        display_within_jupyter: bool = False,
        visualizer: BaseViz = ChartVisualizer,
        movie_name: str = None,
        movie_dir: str = "rddl_movies",
        max_frames=1000,
        enforce_action_constraints=True,
        **kwargs
    ):
        super().__init__(
            rddl_domain=rddl_domain,
            rddl_instance=rddl_instance,
            base_class=base_class,
            backend=backend,
            display_with_pygame=display_with_pygame,
            display_within_jupyter=display_within_jupyter,
            visualizer=visualizer,
            movie_name=movie_name,
            movie_dir=movie_dir,
            max_frames=max_frames,
            enforce_action_constraints=enforce_action_constraints,
            **kwargs
        )


class RDDLDomainSimplifiedSpaces(RDDLDomainRL):
    def _state_step(
        self, action: D.T_event
    ) -> TransitionOutcome[D.T_state, Value[D.T_value], D.T_predicate, D.T_info]:
        outcome = super()._state_step(action)
        return TransitionOutcome(
            state=flatten(self.rddl_gym_env.observation_space, outcome.state),
            value=outcome.value,
            termination=outcome.termination,
        )

    def _get_action_space_(self) -> Space[D.T_event]:
        return GymSpace(flatten_space(self.rddl_gym_env.action_space))

    def _state_reset(self) -> D.T_state:
        # SkDecide only needs the state, not the info
        return flatten(self.rddl_gym_env.observation_space, super()._state_reset())

    def _get_observation_space_(self) -> Space[D.T_observation]:
        return GymSpace(flatten_space(self.rddl_gym_env.observation_space))
