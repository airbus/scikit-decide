import math
import os
import tempfile
from enum import Enum
from typing import NamedTuple

import pytest
from openevolve.config import Config

from skdecide import EnumerableSpace, rollout
from skdecide.builders.domain import FullyObservable, Goals, UnrestrictedActions
from skdecide.hub.domain.maze import Maze
from skdecide.hub.domain.maze.maze import Action
from skdecide.hub.solver.openevolve import IntegratedOpenEvolve, ProxyOpenEvolve


@pytest.fixture
def api_key():
    if "OPENAI_API_KEY" not in os.environ:
        # For demo purposes. Should be exported with proper value before running the script
        os.environ["OPENAI_API_KEY"] = ""


@pytest.fixture
def nb_iterations() -> int:
    """No iteration for github workflow (no easy access to relevant LLM)."""
    return 0


test_dir = f"{os.path.dirname(os.path.abspath(__file__))}"


def _test_proxy_openevolve_maze(domain_cls, solver_kwargs, maze_maps, input_dir):
    initial_program_path = f"{input_dir}/initial_program.py"
    broken_program_path = f"{input_dir}/initial_program_broken.py"
    dfs_program_path = f"{input_dir}/dfs_program.py"

    # Use a temporary directory as save dir
    with tempfile.TemporaryDirectory() as save_dir:
        # solve from working init program
        solver_kwargs["initial_program_path"] = initial_program_path
        with ProxyOpenEvolve(**solver_kwargs) as solver:
            assert solver.using_applicable_actions() is (
                not issubclass(domain_cls, UnrestrictedActions)
            )
            code = solver.get_best_program_code()
            init_res = solver.evaluate_program_code(code)

            solver.solve()  # (no iteration though)

            best_code = solver.get_best_program_code()
            best_res = solver.evaluate_program_code(best_code)

            # dfs > initial program
            with open(dfs_program_path, "r") as f:
                code = f.read()
            dfs_res = solver.evaluate_program_code(code)
            assert (
                dfs_res.metrics["combined_score"] > init_res.metrics["combined_score"]
            )

            # broken prog => -inf score + error + traceback
            with open(broken_program_path, "r") as f:
                code = f.read()
            broken_res = solver.evaluate_program_code(code)
            assert broken_res.metrics["combined_score"] == -math.inf
            assert "traceback" in broken_res.artifacts
            assert "NameError" in broken_res.artifacts["error"]

            # store best prog (and history)
            solver.save(save_dir)

        # load best prog (with solver init from broken prog)
        solver_kwargs["initial_program_path"] = broken_program_path
        with ProxyOpenEvolve(**solver_kwargs) as solver:
            # before restoring previous solve => broken
            broken_res_2 = solver.evaluate_program_code(solver.get_best_program_code())
            assert broken_res_2.metrics["combined_score"] == -math.inf
            # restore
            solver.load(save_dir)
            # best_code now works (from previous solve)
            best_code_2 = solver.get_best_program_code()
            assert best_code_2 == best_code
            best_res_2 = solver.evaluate_program_code(best_code_2)
            assert math.isfinite(best_res_2.metrics["combined_score"])

            # sample prompt
            prompt = solver.sample_prompt()
            assert "system" in prompt
            assert "user" in prompt

            # rollout
            for maze_map in maze_maps:
                rollout_domain = domain_cls(maze_map)
                rollout(
                    solver=solver,
                    domain=rollout_domain,
                    render=False,
                    verbose=False,
                    max_steps=10,
                )


def test_proxy_openevolve_unrestricted_actions(api_key, nb_iterations, maze_maps):
    domain_cls = Maze
    domain_factory = domain_cls
    input_dir = f"{test_dir}/proxy_openevolve_unrestricted_inputs"
    evaluator_path = f"{input_dir}/evaluator.py"

    def planner_kwargs_factory(domain: Maze):
        return dict(goal=domain._goal, maze=domain._maze)

    def planner_action_converter(planner_action: Enum) -> Action:
        return Action[planner_action.name]

    config = Config()
    config.evaluator.cascade_evaluation = False

    solver_kwargs = dict(
        domain_factory=domain_factory,
        evaluator_path=evaluator_path,
        config=config,
        planner_kwargs_factory=planner_kwargs_factory,
        planner_action_converter=planner_action_converter,
        nb_iterations=nb_iterations,
    )

    _test_proxy_openevolve_maze(
        domain_cls=domain_cls,
        solver_kwargs=solver_kwargs,
        maze_maps=maze_maps,
        input_dir=input_dir,
    )


def test_proxy_openevolve_restricted_actions(
    api_key, nb_iterations, restricted_maze_cls, maze_maps
):
    domain_cls = restricted_maze_cls
    domain_factory = domain_cls
    input_dir = f"{test_dir}/proxy_openevolve_restricted_inputs"
    evaluator_path = f"{input_dir}/evaluator.py"

    def planner_kwargs_factory(domain):
        return dict(goal=domain._hub_maze._goal, maze=domain._hub_maze._maze)

    def planner_obs_converter(obs: NamedTuple) -> tuple[int, int]:
        return tuple(obs)

    def planner_action_converter(planner_action: Enum) -> Action:
        return Action[planner_action.name]

    def planner_applicable_actions_converter(
        applicable_actions: EnumerableSpace[Action],
    ) -> list[str]:
        return [a.name for a in applicable_actions.get_elements()]

    config = Config()
    config.evaluator.cascade_evaluation = False

    solver_kwargs = dict(
        domain_factory=domain_factory,
        evaluator_path=evaluator_path,
        config=config,
        planner_kwargs_factory=planner_kwargs_factory,
        planner_obs_converter=planner_obs_converter,
        planner_action_converter=planner_action_converter,
        planner_applicable_actions_converter=planner_applicable_actions_converter,
        nb_iterations=nb_iterations,
    )

    _test_proxy_openevolve_maze(
        domain_cls=domain_cls,
        solver_kwargs=solver_kwargs,
        maze_maps=maze_maps,
        input_dir=input_dir,
    )


@pytest.mark.parametrize(
    "evaluator_enforce_using_public_api, initial_program_include_rollout, prompt_add_blockevolve_instruction, prompt_add_public_api_instruction, prompt_include_domain_module, prompt_include_public_api",
    [
        (True, True, True, True, False, True),
        (False, False, False, False, True, False),
    ],
)
def test_integrated_openevolve_maze(
    maze_cls,
    api_key,
    nb_iterations,
    maze_maps,
    evaluator_enforce_using_public_api,
    initial_program_include_rollout,
    prompt_include_public_api,
    prompt_add_public_api_instruction,
    prompt_add_blockevolve_instruction,
    prompt_include_domain_module,
):
    program_examples_dir = f"{test_dir}/integrated_openevolve_maze_program_examples"
    domain_cls = maze_cls
    evaluator_domain_factories = [
        lambda maze_str=maze_str: domain_cls(maze_str=maze_str)
        for maze_str in maze_maps
    ]

    def evaluator_rollout_max_steps(domain: maze_cls) -> int:
        return 2 * domain.n_cells

    domain_factory = domain_cls
    config = Config()
    config.evaluator.cascade_evaluation = False
    with tempfile.TemporaryDirectory() as output_dir:
        solver_kwargs = dict(
            domain_factory=domain_factory,
            output_dir=output_dir,
            config=config,
            evaluator_domain_factories=evaluator_domain_factories,
            evaluator_rollout_max_steps=evaluator_rollout_max_steps,
            evaluator_enforce_using_public_api=evaluator_enforce_using_public_api,
            initial_program_include_rollout=initial_program_include_rollout,
            prompt_include_public_api=prompt_include_public_api,
            prompt_include_domain_module=prompt_include_domain_module,
            prompt_add_public_api_instruction=prompt_add_public_api_instruction,
            prompt_add_blockevolve_instruction=prompt_add_blockevolve_instruction,
        )

        private_api_program_path = (
            f"{program_examples_dir}/program_using_private_api.py"
        )
        dfs_program_path = f"{program_examples_dir}/dfs_program.py"

        solver = IntegratedOpenEvolve(**solver_kwargs)

        # check built initial program
        initial_code = solver.get_best_program_code()
        exec(initial_code, globals())
        assert (
            'if __name__ == "__main__":' in initial_code
        ) == initial_program_include_rollout

        # test evaluator
        ## initial generated program
        init_res = solver.evaluate_program_code(initial_code)
        assert math.isfinite(init_res.metrics["combined_score"])
        assert ("nb_reached_goals" in init_res.metrics) == issubclass(domain_cls, Goals)

        if issubclass(domain_cls, FullyObservable):
            ## dfs
            with open(dfs_program_path, "r") as f:
                code = f.read()
            dfs_res = solver.evaluate_program_code(code)
            if issubclass(domain_cls, Goals):
                assert (
                    dfs_res.metrics["combined_score"]
                    > init_res.metrics["combined_score"]
                )
            else:
                assert dfs_res.metrics["combined_score"] == -math.inf
                assert "traceback" in dfs_res.artifacts
                assert "AttributeError" in dfs_res.artifacts["error"]
            ## prog using private api
            with open(private_api_program_path, "r") as f:
                code = f.read()
            private_api_res = solver.evaluate_program_code(code)
            if evaluator_enforce_using_public_api:
                assert private_api_res.metrics["combined_score"] == -math.inf
                assert "traceback" in private_api_res.artifacts
                assert "AttributeError" in private_api_res.artifacts["error"]
                assert "Access to private member" in private_api_res.artifacts["error"]
            else:
                assert math.isfinite(private_api_res.metrics["combined_score"])

        # test prompt sampling
        prompt = solver.sample_prompt()
        assert ("# API Reference" in prompt["system"]) == prompt_include_public_api

        # store/load
        with tempfile.TemporaryDirectory() as save_dir:
            solver.save(save_dir)
            solver.load(save_dir)

        # rollout
        for maze_map in maze_maps:
            rollout_domain = domain_cls(maze_map)
            rollout(
                solver=solver,
                domain=rollout_domain,
                render=False,
                verbose=False,
                max_steps=10,
            )


def test_prompt_update_function():
    config = Config()
    config.prompt.system_message = "dummy"

    solver = IntegratedOpenEvolve(
        domain_factory=Maze,
        prompt_include_public_api=False,
        config=config,
    )
    assert "dummy" in solver.sample_prompt()["system"]

    solver = IntegratedOpenEvolve(
        domain_factory=Maze,
        prompt_include_public_api=False,
        prompt_update_function=lambda msg: msg.replace("dummy", "foo"),
        config=config,
    )
    assert "dummy" not in solver.sample_prompt()["system"]
