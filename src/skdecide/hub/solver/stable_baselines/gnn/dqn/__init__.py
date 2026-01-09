from .dqn import GraphDQN as GraphDQN
from .policies import GNNDQNPolicy, MultiInputGNNDQNPolicy

GraphInputPolicy = GNNDQNPolicy
MultiInputPolicy = MultiInputGNNDQNPolicy
