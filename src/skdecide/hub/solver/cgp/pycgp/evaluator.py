# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class Evaluator:
    def evaluate(self, cgp, it):
        raise NotImplementedError("evaluation method not implemented")

    def clone(self):
        raise NotImplementedError("clone method not implemented")
