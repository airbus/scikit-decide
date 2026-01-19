import importlib
import os
import sys

from pytest_cases import fixture, param_fixture

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


@fixture
def restricted_maze_cls():
    module_name = "restricted_action_maze"
    module_path = f"{os.path.dirname(__file__)}/{module_name}.py"

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module

    return vars(module)["Maze"]


@fixture
def maze_maps() -> list[str]:
    with open(f"{os.path.dirname(__file__)}/maze_maps.txt", "r") as f:
        return eval(f.read())


restricted = param_fixture("restricted", [True, False])
goals = param_fixture("goals", [True, False])
partiallyobs = param_fixture("partiallyobs", [True, False])


@fixture
def maze_cls(restricted, goals, partiallyobs):
    if restricted:
        if partiallyobs:
            from restricted_action_obs_maze import Maze, NoGoalMaze
        else:
            from restricted_action_maze import Maze, NoGoalMaze
    else:
        if partiallyobs:
            from unrestricted_action_obs_maze import Maze, NoGoalMaze
        else:
            from unrestricted_action_maze import Maze, NoGoalMaze

    if goals:
        return Maze
    else:
        return NoGoalMaze
