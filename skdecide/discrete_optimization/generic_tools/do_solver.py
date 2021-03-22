# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from skdecide.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage


class SolverDO:
    @abstractmethod
    def solve(self, **kwargs)->ResultStorage:
        ...
