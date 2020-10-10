# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pprint import pprint

from skdecide.utils import get_registered_domains, get_registered_solvers, load_registered_domain, load_registered_solver

if __name__ == '__main__':

    print('\nAll registered domains:\n-----------------------')
    pprint({d: load_registered_domain(d) for d in get_registered_domains()})

    print('\nAll registered solvers:\n-----------------------')
    pprint({s: load_registered_solver(s) for s in get_registered_solvers()})
