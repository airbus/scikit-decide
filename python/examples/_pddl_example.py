import getopt, sys, os

from airlaps.catalog.domain.pddl import *

if __name__ == '__main__':

    try:
        options, remainder = getopt.getopt(sys.argv[1:],
                                           "d:p:l:",
                                           ["domain=", "problem=", "debug_logs="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    domain = 'car_nodrag.pddl'
    problem = ''
    debug_logs = False

    for opt, arg in options:
        if opt in ('-d', '--domain'):
            domain = arg
        elif opt in ('-p', '--problem'):
            problem = arg
        elif opt in ('-l', '--debug_logs'):
            debug_logs = True if arg == 'yes' else False
    
    try:
        mypddl = PDDL()
        mypddl.load(domain, problem, debug_logs)
        print(mypddl.get_domain())
    except RuntimeError as err:
        print(err)