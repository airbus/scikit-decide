# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from skdecide.hub.domain.scheduling.do_to_sk_binding import build_sk_domain


def load_domain(file_path):
    """"""
    from discrete_optimization.rcpsp.parser import parse_file

    rcpsp_model = parse_file(file_path)
    return build_sk_domain(rcpsp_model)


def load_multiskill_domain(file_path):
    from discrete_optimization.rcpsp_multiskill.parser_imopse import parse_file

    mrcpsp_model, new_tame_to_original_task_id = parse_file(file_path)
    return build_sk_domain(mrcpsp_model)
