# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Union

from skdecide.hub.solver.graph_explorer.GraphDomain import (
    GraphDomain,
    GraphDomainUncertain,
)


class GraphExploration:
    @abstractmethod
    def build_graph_domain(
        self, init_state: Any = None
    ) -> Union[GraphDomain, GraphDomainUncertain]:
        ...
