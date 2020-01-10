from __future__ import annotations

import pytest

from skdecide import *
from skdecide.builders.domain import *
from skdecide.core import autocast


# FIXTURES
@pytest.fixture(params=['state', 123, 1.23, ('state', 123, 1.23), ['state', 123, 1.23], {'state_A': 'state', 'state_B': 123, 'state_C': 1.23}])
def state(request):
    return request.param


@pytest.fixture(params=['action', 321, 3.21, ('action', 321, 3.21), ['action', 321, 3.21], {'action_A': 'action', 'action_B': 321, 'action_C': 3.21}])
def action(request):
    return request.param


@pytest.fixture
def general_domain():

    class D(Domain):
        pass

    return D


@pytest.fixture
def specialized_domain():

    class D(Domain, SingleAgent, Sequential, Constrained, DeterministicTransitions, UnrestrictedActions, Goals, DeterministicInitialized, Memoryless, FullyObservable, Renderable, PositiveCosts):
        pass

    return D


# SAMPLE FUNCTIONS
def func_in(memory: D.T_memory[D.T_state], action: D.T_agent[D.T_concurrency[D.T_event]]):
    return memory, action


def func_out1(memory) -> D.T_memory[D.T_state]:
    return memory


def func_out2(action) -> D.T_agent[D.T_concurrency[D.T_event]]:
    return action


# TESTS
def test_down_cast_in(general_domain, specialized_domain, state, action):
    cast = autocast(func_in, general_domain, specialized_domain)
    assert cast(state, action) == (Memory([state]), {'agent': [action]})


def test_up_cast_in(specialized_domain, general_domain, state, action):
    cast = autocast(func_in, specialized_domain, general_domain)
    assert cast(Memory([state]), {'agent': [action]}) == (state, action)


def test_down_cast_out1(general_domain, specialized_domain, state):
    cast = autocast(func_out1, general_domain, specialized_domain)
    assert cast(Memory([state])) == state


def test_down_cast_out2(general_domain, specialized_domain, action):
    cast = autocast(func_out2, general_domain, specialized_domain)
    assert cast({'agent': [action]}) == action


def test_up_cast_out1(specialized_domain, general_domain, state):
    cast = autocast(func_out1, specialized_domain, general_domain)
    assert cast(state) == Memory([state])


def test_up_cast_out2(specialized_domain, general_domain, action):
    cast = autocast(func_out2, specialized_domain, general_domain)
    assert cast(action) == {'agent': [action]}
