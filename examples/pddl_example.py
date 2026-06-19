# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import getopt
import sys

from skdecide.hub.domain.pddl import *

if __name__ == "__main__":
    try:
        options, remainder = getopt.getopt(
            sys.argv[1:], "d:p:l:", ["domain=", "problem=", "debug_logs="]
        )
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    domain = "car_nodrag.pddl"
    problem = ""
    debug_logs = False

    for opt, arg in options:
        if opt in ("-d", "--domain"):
            domain = arg
        elif opt in ("-p", "--problem"):
            problem = arg
        elif opt in ("-l", "--debug_logs"):
            debug_logs = True if arg == "yes" else False

    try:
        mypddl = PDDL()
        mypddl.load([domain], debug_logs)
        for d in mypddl.get_domains():
            print(d)
        for p in mypddl.get_problems():
            print(p)
    except RuntimeError as err:
        print(err)
