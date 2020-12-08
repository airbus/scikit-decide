from typing import Any, Union
from abc import abstractmethod
from skdecide.hub.solver.graph_explorer.GraphDomain import GraphDomain, GraphDomainUncertain


class GraphExploration:
    @abstractmethod
    def build_graph_domain(self, init_state: Any=None)->Union[GraphDomain, GraphDomainUncertain]:
        ...
