# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import re

import numpy as np
import pytest

from skdecide.hub.solver.openevolve.api_extraction import (
    ApiExtractionParams,
    generate_public_api,
)


@pytest.mark.parametrize(
    "params",
    [
        None,
        ApiExtractionParams(
            recursive=True,
            user_modules=[np],
            include_hub_api=False,
            include_domain_cls_module=True,
            extract_observation_space_cls=False,
            extract_action_space_cls=True,
        ),
        ApiExtractionParams(
            recursive=True,
            include_domain_cls_module=False,
            include_hub_api=True,
            subtitle_level=5,
            extract_action_space_cls=False,
            extract_observation_space_cls=True,
            simplify_signature=False,
            strip_admonitions=False,
        ),
        ApiExtractionParams(
            recursive=False,
            strip_admonitions=True,
        ),
    ],
)
def test_generate_public_api(params, restricted_maze_cls):
    Maze = restricted_maze_cls
    domain_cls = Maze
    domain = domain_cls()
    doc = generate_public_api(cls=domain_cls, domain=domain, params=params)
    if params is None:
        params = ApiExtractionParams()

    subtitle_prefix = "#" * params.subtitle_level
    assert re.search(
        rf"^{subtitle_prefix} API Reference:.*Maze.*", doc, flags=re.MULTILINE
    )
    assert re.search(r"Domain Capabilities", doc)
    assert re.search(r"Agent.*:.*SingleAgent", doc)
    assert re.search(r"Domain base types", doc)
    assert re.search(r"T_observation.*:.*State", doc)
    assert (
        bool(re.search(r"Action space per agent", doc))
        is params.extract_action_space_cls
    )
    assert (
        bool(re.search(r"Observation space per agent", doc))
        is params.extract_observation_space_cls
    )
    assert re.search(r"Attributes.*\n.*n_cells", doc)
    assert re.search(r"check_value\(self, value:", doc)
    assert bool(re.search(r"\*\*WARNING:\*\*", doc)) is (not params.strip_admonitions)
    assert bool(re.search(r"D.T_agent", doc)) is (not params.simplify_signature)
    assert bool(re.search(r"API Reference:.*Action.*", doc)) is (
        params.include_hub_api and params.recursive
    )
    assert bool(re.search(r"API Reference:.*Dummy.*", doc)) is (
        params.include_domain_cls_module and params.recursive
    )
    assert bool(re.search(r"API Reference:.*numpy.ndarray.*", doc)) is (
        (np in params.user_modules) and params.recursive
    )
