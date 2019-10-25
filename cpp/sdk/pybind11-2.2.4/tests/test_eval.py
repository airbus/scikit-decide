# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pybind11_tests import eval_ as m


def test_evals(capture):
    with capture:
        assert m.test_eval_statements()
    assert capture == "Hello World!"

    assert m.test_eval()
    assert m.test_eval_single_statement()

    filename = os.path.join(os.path.dirname(__file__), "test_eval_call.py")
    assert m.test_eval_file(filename)

    assert m.test_eval_failure()
    assert m.test_eval_file_failure()
