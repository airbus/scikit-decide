from typing import Sequence

from skdecide.core import EnumerableSpace
from skdecide.hub.space.gym import ListSpace


class MyListSpace(EnumerableSpace[int]):
    def __init__(self, the_list: list[int]):
        self.the_list = the_list

    def get_elements(self) -> Sequence[int]:
        return self.the_list


def test_my_enumerable_space():
    space = MyListSpace([1, -2, 3])
    assert -2 in space
    assert space[2] == 3
    assert list(reversed(space)) == [3, -2, 1]
    assert list(space) == space.the_list
    for e in space:
        assert isinstance(e, int)


def test_gym_listspace():
    space = ListSpace([1, -2, 3])
    assert -2 in space
    assert space[2] == 3
    assert list(reversed(space)) == [3, -2, 1]
    assert list(space) == space.get_elements()
    for e in space:
        assert isinstance(e, int)
