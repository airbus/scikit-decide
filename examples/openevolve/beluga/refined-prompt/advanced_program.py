import random

from skd_domains.skd_pddl_domain import SkdPDDLDomain

from skdecide import Space


# EVOLVE-BLOCK-START
class Planner:
    """Heuristic sampler for the Beluga logistics domain.

    This planner uses serialized PDDL states to prioritize:
    1. Mandatory sequences (to_unload, to_deliver, to_load).
    2. Workflow progression (Beluga -> Rack -> Hangar -> Beluga).
    3. Avoiding high-cost unblocking (swaps) unless necessary.
    """

    def __init__(self, domain: SkdPDDLDomain):
        self.domain = domain

    def sample_action(
        self,
        state: SkdPDDLDomain.T_state,
        applicable_actions: Space[SkdPDDLDomain.T_event],
    ) -> SkdPDDLDomain.T_event:
        actions_list = list(applicable_actions.get_elements())
        if not actions_list:
            return None
        if len(actions_list) == 1:
            return actions_list[0]

        # 1. Bridge the compact state to readable PDDL strings
        readable_atoms, readable_fluents = self.domain.serialize_state(state)

        # Priority Buckets
        critical = []  # Direct mission goals (unload/load/deliver)
        progress = []  # Transitions (moving between racks and trailers)
        maintenance = []  # Unblocking/Swapping

        for action in actions_list:
            # 2. Bridge the action to readable PDDL name and parameters
            serialized = self.domain.serialize_action(action)
            action_name = serialized[0]

            # --- Tier 1: Critical Workflow Actions ---
            # Check for to_unload or to_load requirements in the serialized atoms
            if action_name in ["unload-beluga", "load-beluga", "beluga-complete"]:
                critical.append(action)
                continue

            # Deliver to hangar is critical if the jig is 'to_deliver'
            if action_name == "deliver-to-hangar":
                jig = serialized[1]
                # Match against (to_deliver, jig, line) in atoms
                if any(
                    atom[0] == "to_deliver" and atom[1] == jig
                    for atom in readable_atoms
                ):
                    critical.append(action)
                    continue

            # --- Tier 2: Progress Actions ---
            # Standard moves to/from racks or hangars
            if action_name in ["put-down-rack", "get-from-hangar"]:
                progress.append(action)
                continue

            # Unstacking from rack is progress if the jig is needed for a goal
            if action_name == "unstack-rack":
                jig = serialized[1]
                is_needed = any(
                    atom[1] == jig and atom[0] in ["to_deliver", "to_load"]
                    for atom in readable_atoms
                )
                if is_needed:
                    progress.append(action)
                else:
                    maintenance.append(action)
                continue

            # --- Tier 3: Maintenance/Swapping ---
            # Stacking or swapping to manage rack space
            if action_name in ["stack-rack", "pick-up-rack"]:
                maintenance.append(action)
            else:
                progress.append(action)

        # Hierarchical Selection
        if critical:
            return random.choice(critical)
        if progress:
            return random.choice(progress)
        if maintenance:
            return random.choice(maintenance)

        return random.choice(actions_list)


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    import json
    import os
    import sys

    from dotenv import load_dotenv

    # load environment variables defining path to beluga file and beluga challenge toolkit
    load_dotenv()

    # "install" toolkit
    beluga_toolkit_repo = os.path.abspath(os.environ["BELUGA_TOOLKIT_REPO"])
    sys.path.append(beluga_toolkit_repo)
    from beluga_lib.beluga_problem import BelugaProblemDecoder
    from skd_domains.skd_pddl_domain import SkdPDDLDomain

    # create domain
    def create_domain_from_json(
        problem_json: str, classic: bool = False
    ) -> SkdPDDLDomain:
        problem_folder = os.path.dirname(problem_json)
        problem_name = os.path.basename(problem_json)
        with open(problem_json, "r") as fp:
            inst = json.load(fp, cls=BelugaProblemDecoder)
        domain = SkdPDDLDomain(inst, problem_name, problem_folder, classic=classic)
        domain.n_jigs = len(inst.jigs)
        return domain

    domain = create_domain_from_json(os.environ["BELUGA_JSON"])

    # instantiate planner for this domain
    planner = Planner(domain=domain)

    max_steps = 20 * domain.n_jigs
    nb_step = 0
    total_cost = 0
    state = domain.get_initial_state()

    while not domain.is_terminal(state) and nb_step < max_steps:
        applicable_actions = domain.get_applicable_actions(state)
        action = planner.sample_action(state, applicable_actions)
        next_state = domain.get_next_state(memory=state, action=action)
        value = domain.get_transition_value(
            memory=state, action=action, next_state=next_state
        )

        state = next_state
        total_cost += value.cost
        nb_step += 1

    print(f"total cost: {total_cost}")
    if domain.is_goal(state):
        print("Goal reached!")
    else:
        print("Goal not reached.")
