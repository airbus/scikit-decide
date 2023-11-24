from skdecide.hub.domain.rock_paper_scissors.rock_paper_scissors import (
    RockPaperScissors,
)
from skdecide.utils import rollout


def test_rock_paper_scissors():
    domain = RockPaperScissors()
    rollout(
        domain,
        action_formatter=lambda a: str({k: v.name for k, v in a.items()}),
        outcome_formatter=lambda o: f"{ {k: v.name for k, v in o.observation.items()} }"
        f" - rewards: { {k: v.reward for k, v in o.value.items()} }",
    )
