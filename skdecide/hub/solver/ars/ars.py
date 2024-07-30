# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import gymnasium as gym
import numpy as np
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
    IntegerHyperparameter,
)

from skdecide import Domain, Solver
from skdecide.builders.domain import (
    Environment,
    History,
    Initializable,
    PartiallyObservable,
    Rewards,
    Sequential,
    SingleAgent,
    UnrestrictedActions,
)
from skdecide.builders.solver import Policies, Restorable
from skdecide.hub.solver.cgp import cgp


class D(
    Domain,
    SingleAgent,
    Sequential,
    Environment,
    UnrestrictedActions,
    Initializable,
    History,
    PartiallyObservable,
    Rewards,
):
    pass


# for normalizing states
class Normalizer:
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        if self.n[0] <= 1:
            return inputs
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


def flatten(c):
    """
    Generator flattening the structure
    """
    for x in c:
        if isinstance(x, str) or not isinstance(x, Iterable):
            yield x
        else:
            yield from flatten(x)


class AugmentedRandomSearch(Solver, Policies):
    """Augmented Random Search solver."""

    T_domain = D

    hyperparameters = [
        IntegerHyperparameter(name="n_epochs"),
        IntegerHyperparameter(name="epoch_size"),
        IntegerHyperparameter(name="directions"),
        IntegerHyperparameter(name="top_directions"),
        FloatHyperparameter(name="learning_rate"),
        FloatHyperparameter(name="policy_noise"),
    ]

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        n_epochs=1000,
        epoch_size=1000,
        directions=10,
        top_directions=3,
        learning_rate=0.02,
        policy_noise=0.03,
        reward_maximization=True,
        callback: Callable[[AugmentedRandomSearch], bool] = lambda solver: False,
    ) -> None:
        """

        # Parameters
        domain_factory
        n_epochs
        epoch_size
        directions
        top_directions
        learning_rate
        policy_noise
        reward_maximization
        callback: function called at each solver epoch. If returning true, the solve process stops.

        """
        self.callback = callback
        Solver.__init__(self, domain_factory=domain_factory)
        self.env = None
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.epoch_size = epoch_size
        self.directions = directions
        self.top_directions = top_directions
        self.policy = None
        self.policy_noise = policy_noise
        self.reward_maximization = reward_maximization
        assert self.top_directions <= self.directions

    def evaluate_policy(self, state, delta=None, direction=None):
        if direction is None:
            return self.policy.dot(state)
        elif direction == "positive":
            return (self.policy + self.policy_noise * delta).dot(state)
        else:
            return (self.policy - self.policy_noise * delta).dot(state)

    def explore(self, normalizer, direction=None, delta=None):
        state = self.env.reset()

        done = False
        num_plays = 0.0
        sum_rewards = 0
        while not done and num_plays < self.epoch_size:

            state = cgp.norm_and_flatten(
                state, self.env.get_observation_space().unwrapped()
            )
            action = self.evaluate_policy(state, delta, direction)

            action = cgp.denorm(action, self.env.get_action_space().unwrapped())
            state, transition_value, done, _ = self.env.step(action).astuple()

            reward = transition_value[0]
            reward = max(min(reward, 1), -1)
            if not np.isnan(reward):
                sum_rewards += reward
            num_plays += 1
        return sum_rewards

    def update_policy(self, rollouts, sigma_r):
        step = np.zeros(self.policy.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        if self.top_directions == 0 or sigma_r == 0:
            return
        self.policy += self.learning_rate / (self.top_directions * sigma_r) * step

    def get_dimension_space(self, space):
        if isinstance(space, gym.spaces.Tuple):
            dim = 0
            for element in space:
                dim += self.get_dimension_space(element)
            return dim
        elif isinstance(space, gym.spaces.Discrete):
            return 1
        else:
            return space.shape[0]

    def generate_perturbations(self, space):
        if isinstance(space, gym.spaces.Tuple):
            perturbations = []
            for element in space:
                perturbations += self.generate_perturbations(element)
            return perturbations
        if isinstance(space, gym.spaces.Discrete):
            return 2 * np.random.random_integers(space.n) / space.n - 1
        else:
            return 2 * np.random.random_sample() - 1

    def _solve(self) -> None:
        self.env = self._domain_factory()
        np.random.seed(0)
        input_size = self.get_dimension_space(
            self.env.get_observation_space().unwrapped()
        )
        output_size = self.get_dimension_space(self.env.get_action_space().unwrapped())

        self.policy = np.zeros((output_size, input_size))

        normalizer = Normalizer(input_size)

        for step in range(self.n_epochs):

            # Initializing the perturbations deltas and the positive/negative rewards

            deltas = [
                2 * np.random.random_sample(self.policy.shape) - 1
                for _ in range(self.directions)
            ]
            positive_rewards = [0] * self.directions
            negative_rewards = [0] * self.directions

            # Getting the positive rewards in the positive directions
            for k in range(self.directions):
                positive_rewards[k] = self.explore(
                    normalizer, direction="positive", delta=deltas[k]
                )

            # Getting the negative rewards in the negative/opposite directions
            for k in range(self.directions):
                negative_rewards[k] = self.explore(
                    normalizer, direction="negative", delta=deltas[k]
                )

            # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
            all_rewards = np.array(positive_rewards + negative_rewards)
            sigma_r = all_rewards.std()

            # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
            scores = {
                k: max(r_pos, r_neg)
                for k, (r_pos, r_neg) in enumerate(
                    zip(positive_rewards, negative_rewards)
                )
            }
            order = sorted(
                scores.keys(), key=lambda x: scores[x], reverse=self.reward_maximization
            )[: self.top_directions]
            rollouts = [
                (positive_rewards[k], negative_rewards[k], deltas[k]) for k in order
            ]

            # Updating our policy
            self.update_policy(rollouts, sigma_r)

            # Printing the final reward of the policy after the update
            self.reward_evaluation = self.explore(normalizer)
            print(
                "Step:", step, "Reward:", self.reward_evaluation, "Policy", self.policy
            )

            # Stopping because of user's callback?
            if self.callback(self):
                break

        print("Final Reward:", self.reward_evaluation, "Policy", self.policy)

    def _sample_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:

        # print('observation', observation, 'Policy', self.policy)
        action = self.policy.dot(
            cgp.norm_and_flatten(
                observation, self.env.get_observation_space().unwrapped()
            )
        )
        action = cgp.denorm(action, self.env.get_action_space().unwrapped())
        return action
