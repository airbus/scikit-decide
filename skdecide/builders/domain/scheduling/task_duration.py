# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict, Optional

from skdecide.core import DiscreteDistribution, Distribution

__all__ = [
    "SimulatedTaskDuration",
    "UncertainMultivariateTaskDuration",
    "UncertainUnivariateTaskDuration",
    "UncertainBoundedTaskDuration",
    "UniformBoundedTaskDuration",
    "EnumerableTaskDuration",
    "DeterministicTaskDuration",
]


class SimulatedTaskDuration:
    """A domain must inherit this class if the task duration requires sampling from a simulation."""

    # TODO, this can be challenged.. for uncertain domain (with adistribution, you want to sample a different value each time.
    # that 's why i override this sample_task_duration in below level.
    def sample_task_duration(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Sample, store and return task duration for the given task in the given mode."""
        if task not in self.sampled_durations:
            self.sampled_durations[task] = {}
        if mode not in self.sampled_durations[task]:
            self.sampled_durations[task][mode] = {}
        if progress_from not in self.sampled_durations[task][mode]:
            self.sampled_durations[task][mode][
                progress_from
            ] = self._sample_task_duration(task, mode, progress_from)
        return self.sampled_durations[task][mode][progress_from]

    def _sample_task_duration(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return a task duration for the given task in the given mode."""
        raise NotImplementedError

    def get_latest_sampled_duration(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ):
        if task in self.sampled_durations:
            if mode in self.sampled_durations[task]:
                if progress_from in self.sampled_durations[task][mode]:
                    return self.sampled_durations[task][mode][progress_from]
        return self.sample_task_duration(task, mode, progress_from)


# TODO: Can we currently model multivariate distribution with the Distribution object ?
class UncertainMultivariateTaskDuration(SimulatedTaskDuration):
    """A domain must inherit this class if the task duration is uncertain and follows a know multivariate
    distribution."""

    def sample_task_duration(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return a task duration for the given task in the given mode,
        sampled from the underlying multiivariate distribution."""
        return self._sample_task_duration(
            task=task, mode=mode, progress_from=progress_from
        )

    def _sample_task_duration(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return a task duration for the given task in the given mode,
        sampled from the underlying multiivariate distribution."""
        return self.get_task_duration_distribution(task, mode).sample()

    def get_task_duration_distribution(
        self,
        task: int,
        mode: Optional[int] = 1,
        progress_from: Optional[float] = 0.0,
        multivariate_settings: Optional[Dict[str, int]] = None,
    ) -> Distribution:
        """Return the multivariate Distribution of the duration of the given task in the given mode.
        Multivariate seetings need to be provided."""
        return self._get_task_duration_distribution(
            task, mode, progress_from, multivariate_settings
        )

    def _get_task_duration_distribution(
        self,
        task: int,
        mode: Optional[int] = 1,
        progress_from: Optional[float] = 0.0,
        multivariate_settings: Optional[Dict[str, int]] = None,
    ) -> Distribution:
        """Return the multivariate Distribution of the duration of the given task in the given mode.
        Multivariate seetings need to be provided."""
        raise NotImplementedError


class UncertainUnivariateTaskDuration(UncertainMultivariateTaskDuration):
    """A domain must inherit this class if the task duration is uncertain and follows a know univariate distribution."""

    def _sample_task_duration(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return a task duration for the given task in the given mode,
        sampled from the underlying univariate distribution."""
        return self.get_task_duration_distribution(task, mode).sample()

    def _get_task_duration_distribution(
        self,
        task: int,
        mode: Optional[int] = 1,
        progress_from: Optional[float] = 0.0,
        multivariate_settings: Optional[Dict[str, int]] = None,
    ) -> Distribution:  # TODO, problem here i think
        """Return the univariate Distribution of the duration of the given task in the given mode."""
        raise NotImplementedError


class UncertainBoundedTaskDuration(UncertainUnivariateTaskDuration):
    """A domain must inherit this class if the task duration is known to be between a lower and upper bound
    and follows a known distribution between these bounds."""

    def _sample_task_duration(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return a task duration for the given task in the given mode,
        sampled from the underlying univariate bounded distribution."""
        return self.get_task_duration_distribution(task, mode).sample()

    def _get_task_duration_distribution(
        self,
        task: int,
        mode: Optional[int] = 1,
        progress_from: Optional[float] = 0.0,
        multivariate_settings: Optional[Dict[str, int]] = None,
    ) -> DiscreteDistribution:
        """Return the Distribution of the duration of the given task in the given mode.
        The distribution returns values beween the defined lower and upper bounds."""
        raise NotImplementedError

    def get_task_duration_upper_bound(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return the upper bound for the task duration of the given task in the given mode."""
        return self._get_task_duration_upper_bound(task, mode, progress_from)

    def _get_task_duration_upper_bound(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return the upper bound for the task duration of the given task in the given mode."""
        raise NotImplementedError

    def get_task_duration_lower_bound(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return the lower bound for the task duration of the given task in the given mode."""
        return self._get_task_duration_lower_bound(task, mode, progress_from)

    def _get_task_duration_lower_bound(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return the lower bound for the task duration of the given task in the given mode."""
        raise NotImplementedError


class UniformBoundedTaskDuration(UncertainBoundedTaskDuration):
    """A domain must inherit this class if the task duration is known to be between a lower and upper bound
    and follows a uniform distribution between these bounds."""

    def _sample_task_duration(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return a task duration for the given task in the given mode,
        sampled from the underlying univariate uniform bounded distribution."""
        return self.get_task_duration_distribution(task, mode).sample()

    def _get_task_duration_distribution(
        self,
        task: int,
        mode: Optional[int] = 1,
        progress_from: Optional[float] = 0.0,
        multivariate_settings: Optional[Dict[str, int]] = None,
    ) -> DiscreteDistribution:
        """Return the Distribution of the duration of the given task in the given mode.
        The distribution is uniform between the defined lower and upper bounds."""
        lb = self.get_task_duration_lower_bound(task, mode)
        ub = self.get_task_duration_upper_bound(task, mode)
        n_vals = ub - lb + 1
        p = 1.0 / float(n_vals)
        values = [(x, p) for x in range(lb, ub + 1)]
        return DiscreteDistribution(values)

    def _get_task_duration_upper_bound(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return the upper bound for the task duration of the given task in the given mode."""
        raise NotImplementedError

    def _get_task_duration_lower_bound(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return the lower bound for the task duration of the given task in the given mode."""
        raise NotImplementedError


class EnumerableTaskDuration(UncertainBoundedTaskDuration):
    """A domain must inherit this class if the task duration for each task is enumerable."""

    def _sample_task_duration(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return a task duration for the given task in the given mode."""
        return self.get_task_duration_distribution(task, mode).sample()

    def _get_task_duration_distribution(
        self,
        task: int,
        mode: Optional[int] = 1,
        progress_from: Optional[float] = 0.0,
        multivariate_settings: Optional[Dict[str, int]] = None,
    ) -> DiscreteDistribution:
        """Return the Distribution of the duration of the given task in the given mode.
        as an Enumerable."""
        raise NotImplementedError

    def _get_task_duration_upper_bound(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return the upper bound for the task duration of the given task in the given mode."""
        duration_vals = [
            x[0] for x in self.get_task_duration_distribution(task, mode).get_values()
        ]
        return max(duration_vals)

    def _get_task_duration_lower_bound(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return the lower bound for the task duration of the given task in the given mode."""
        duration_vals = [
            x[0] for x in self.get_task_duration_distribution(task, mode).get_values()
        ]
        return min(duration_vals)


class DeterministicTaskDuration(EnumerableTaskDuration):
    """A domain must inherit this class if the task durations are known and deterministic."""

    def _sample_task_duration(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return a task duration for the given task in the given mode."""
        return self.get_task_duration(task, mode, progress_from)

    def get_task_duration(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return the fixed deterministic task duration of the given task in the given mode."""
        return self._get_task_duration(task, mode, progress_from)

    def _get_task_duration(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return the fixed deterministic task duration of the given task in the given mode."""
        raise NotImplementedError

    def _get_task_duration_distribution(
        self,
        task: int,
        mode: Optional[int] = 1,
        progress_from: Optional[float] = 0.0,
        multivariate_settings: Optional[Dict[str, int]] = None,
    ):
        """Return the Distribution of the duration of the given task in the given mode.
        Because the duration is deterministic, the distribution always returns the same duration."""
        return DiscreteDistribution([(self.get_task_duration(task, mode), 1)])

    def _get_task_duration_upper_bound(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return the upper bound for the task duration of the given task in the given mode."""
        return self.get_task_duration(task, mode)

    def _get_task_duration_lower_bound(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        """Return the lower bound for the task duration of the given task in the given mode."""
        return self.get_task_duration(task, mode)
