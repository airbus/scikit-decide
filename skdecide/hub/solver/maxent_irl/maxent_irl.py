# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Callable

import gymnasium as gym
import numpy as np
import pylab
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
    IntegerHyperparameter,
)

from skdecide import D, Domain, RLDomain, Solver
from skdecide.builders.solver import Policies, Restorable
from skdecide.hub.solver.cgp import cgp


class D(RLDomain):
    pass


class MaxentIRL(Solver, Policies):
    """Maximum Entropy Inverse Reinforcement Learning solver."""

    T_domain = D

    hyperparameters = [
        IntegerHyperparameter(name="n_states"),
        IntegerHyperparameter(name="n_actions"),
        IntegerHyperparameter(name="one_feature"),
        FloatHyperparameter(name="gamma"),
        FloatHyperparameter(name="q_learning_rate"),
        FloatHyperparameter(name="theta_learning_rate"),
        IntegerHyperparameter(name="n_epochs"),
    ]

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        n_states=400,
        n_actions=3,
        one_feature=20,
        gamma=0.99,
        q_learning_rate=0.03,
        theta_learning_rate=0.05,
        n_epochs=20000,
        expert_trajectories="maxent_expert_demo.npy",
        callback: Callable[[MaxentIRL], bool] = lambda solver: False,
    ) -> None:
        """

        # Parameters
        domain_factory
        n_states
        n_actions
        one_feature
        gamma
        q_learning_rate
        theta_learning_rate
        n_epochs
        expert_trajectories
        callback: function called at each solver epoch. If returning true, the solve process stops.

        """
        self.callback = callback
        Solver.__init__(self, domain_factory=domain_factory)
        self.n_states = n_states
        self.feature_matrix = np.eye(self.n_states)
        self.n_actions = n_actions
        self.one_feature = one_feature

        self.q_table = np.zeros((n_states, n_actions))
        self.env = None
        self.state = None
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.q_learning_rate = q_learning_rate
        self.theta_learning_rate = theta_learning_rate
        self.expert_trajectories = expert_trajectories

        self.score = 0
        if os.path.isfile(expert_trajectories[:-4] + "_maxent_q_table.npy"):
            self.q_table = np.load(
                file=expert_trajectories[:-4] + "_maxent_q_table.npy"
            )
        else:
            self.q_table = None

    def expert_feature_expectations(self, demonstrations):
        feature_expectations = np.zeros(self.feature_matrix.shape[0])

        for demonstration in demonstrations:
            for state_index, _, _ in demonstration:
                feature_expectations += self.feature_matrix[int(state_index)]

        feature_expectations /= demonstrations.shape[0]
        return feature_expectations

    def state_to_index(self, observation_space, state):
        env_low = observation_space.low
        env_high = observation_space.high
        env_distance = (env_high - env_low) / self.one_feature
        position_idx = int((state[0] - env_low[0]) / env_distance[0])
        velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
        state_index = position_idx + velocity_idx * self.one_feature
        return state_index

    def index_to_state(self, observation_space, state):
        """Convert pos and vel about mounting car environment to the integer value"""
        env_low = observation_space.low
        env_high = observation_space.high
        env_distance = (env_high - env_low) / self.one_feature
        position_idx = int((state[0] - env_low[0]) / env_distance[0])
        velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
        state_idx = position_idx + velocity_idx * self.one_feature
        return state_idx

    def gradient_vector(self, expert, learner, theta):
        gradient = expert - learner
        theta += self.theta_learning_rate * gradient
        # Clip theta
        for j in range(len(theta)):
            if theta[j] > 0:
                theta[j] = 0

    def adapt_action_to_environment(self, vals, types):

        if not isinstance(types, Iterable) and not isinstance(types, gym.spaces.Tuple):
            types = [types]
        if not isinstance(vals, Iterable) and not isinstance(vals, gym.spaces.Tuple):
            vals = [vals]
        out = []
        index = 0
        for i in range(len(types)):
            t = types[i]
            if isinstance(t, gym.spaces.Box):
                out_temp = []
                for j in range(len(t.low)):
                    out_temp += [
                        cgp.change_interval(vals[index], -1, 1, t.low[j], t.high[j])
                    ]
                    index += 1
                out += [out_temp]
            elif isinstance(t, gym.spaces.Discrete):
                out += [vals[index] % t.n]
                index += 1
            else:
                raise ValueError("Unsupported type ", str(t))
        # burk
        if len(types) == 1:
            return out[0]
        else:
            return out

    def update_q_table(self, state, action, reward, next_state):
        q_1 = self.q_table[state][action]
        q_2 = reward + self.gamma * max(self.q_table[next_state])
        self.q_table[state][action] += self.q_learning_rate * (q_2 - q_1)

    def process_stable_baselines_npz_trajs(self, rawFile):
        traj_length = rawFile["episode_returns"].max
        num_traj = rawFile["episode_returns"].size
        demonstrations = np.zeros((num_traj, traj_length, rawFile["obs"].shape[2]))
        index = 0
        for x in range(num_traj):
            for y in range(traj_length):
                demonstrations[x][y][0] = index
                demonstrations[x][y][1] = rawFile["actions"][index]
                index += 1

    def _solve(self) -> None:
        self.env = self._domain_factory()
        if self.q_table is not None:
            return
        self.q_table = np.zeros((self.n_states, self.n_actions))
        # raw_demo2 = np.load(file=self.expert_trajectories)
        # if isinstance(raw_demo, np.lib.npyio.NpzFile):

        if not os.path.isfile(self.expert_trajectories):
            print("file with expert trajectories not found")
            return
        if not isinstance(self.env.get_action_space().unwrapped(), gym.spaces.Discrete):
            print(
                "Warning! This IRL method relies on Q-tables so the action space is treated as discrete"
            )
        raw_demo = np.load(file=self.expert_trajectories)
        demonstrations = np.zeros(
            (raw_demo.shape[0], raw_demo.shape[1], raw_demo.shape[2])
        )
        env_low = self.env.get_observation_space().unwrapped().low
        env_high = self.env.get_observation_space().unwrapped().high
        env_distance = (env_high - env_low) / self.one_feature
        for x in range(len(raw_demo)):
            for y in range(len(raw_demo[0])):
                position_idx = int((raw_demo[x][y][0] - env_low[0]) / env_distance[0])
                velocity_idx = int((raw_demo[x][y][1] - env_low[1]) / env_distance[1])
                state_index = position_idx + velocity_idx * self.one_feature

                demonstrations[x][y][0] = state_index
                demonstrations[x][y][1] = raw_demo[x][y][2]

        np.random.seed(1)
        expert = self.expert_feature_expectations(demonstrations)
        learner_feature_expectations = np.zeros(self.n_states)

        theta = -(np.random.uniform(size=(self.n_states,)))

        episodes, scores = [], []

        for episode in range(self.n_epochs):
            state = self.env.reset()
            score = 0

            if (episode != 0 and episode == 10000) or (
                episode > 10000 and episode % 5000 == 0
            ):
                learner = learner_feature_expectations / episode
                self.gradient_vector(expert, learner, theta)

            while True:
                state_index = self.state_to_index(
                    self.env.get_observation_space().unwrapped(), state
                )
                action = np.argmax(self.q_table[state_index])
                next_state, transition_value, done, _ = self.env.step(action).astuple()
                reward = transition_value[0]  # TODO: correct Gym wrapper

                irl_rewards = self.feature_matrix.dot(theta).reshape((self.n_states,))
                irl_reward = irl_rewards[state_index]

                next_state_index = self.state_to_index(
                    self.env.get_observation_space().unwrapped(), next_state
                )

                q_1 = self.q_table[state_index][action]
                q_2 = irl_reward + self.gamma * max(self.q_table[next_state_index])
                self.q_table[state_index][action] += self.q_learning_rate * (q_2 - q_1)

                learner_feature_expectations += self.feature_matrix[int(state_index)]

                score += reward
                state = next_state

                if done:
                    scores.append(score)
                    episodes.append(episode)
                    break

            if episode % 1000 == 0:
                score_avg = np.mean(scores)
                print("{} episode score is {:.2f}".format(episode, score_avg))
                pylab.plot(episodes, scores, "b")
                pylab.savefig("./maxent.png")
                np.save(
                    "./" + self.expert_trajectories[:-4] + "_maxent_q_table",
                    arr=self.q_table,
                )

            # Stopping because of user's callback?
            if self.callback(self):
                break

        self.q_table = np.load(
            file=self.expert_trajectories[:-4] + "_maxent_q_table.npy"
        )

    def _sample_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:

        state_idx = self.index_to_state(
            self.env.get_observation_space().unwrapped(), observation
        )
        action = np.argmax(self.q_table[state_idx])
        return self.adapt_action_to_environment(
            action, self.env.get_action_space().unwrapped()
        )

    def _reset(self) -> None:
        self.state = self.env.reset()
        self.score = 0

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        pass
