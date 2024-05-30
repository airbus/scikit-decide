from os.path import dirname

from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_domain

path_to_data = f"{dirname(__file__)}/../../../examples/scheduling/data"
path_to_data_rcpsp = f"{path_to_data}/rcpsp"


def test_rcpsp_sk():
    domain = load_domain(f"{path_to_data_rcpsp}/j1201_1.sm")
    state = domain.get_initial_state()
    print("Initial state : ", state)
    actions = domain.get_applicable_actions(state)
    print([str(action) for action in actions.get_elements()])
    action = actions.get_elements()[0]
    new_state = domain.get_next_state(state, action)
    print("New state ", new_state)
    actions = domain.get_applicable_actions(new_state)
    print("New actions : ", [str(action) for action in actions.get_elements()])
    action = actions.get_elements()[0]
    print(action)
    new_state = domain.get_next_state(new_state, action)
    print("New state :", new_state)
    print("_is_terminal: ", domain._is_terminal(state))
