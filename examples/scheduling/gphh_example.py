import json
import operator
import os
import pickle

import numpy as np

from examples.scheduling.rcpsp_datasets import get_complete_path, get_data_available
from skdecide import rollout_episode
from skdecide.hub.domain.rcpsp.rcpsp_sk import RCPSP
from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_domain
from skdecide.hub.solver.do_solver.do_solver_scheduling import DOSolver, SolvingMethod
from skdecide.hub.solver.gphh.gphh import (
    GPHH,
    EvaluationGPHH,
    FeatureEnum,
    GPHHPolicy,
    ParametersGPHH,
    PermutationDistance,
    PoolAggregationMethod,
    PooledGPHHPolicy,
    PrimitiveSet,
    feature_average_resource_requirements,
    feature_n_predecessors,
    feature_n_successors,
    feature_task_duration,
    feature_total_n_res,
    max_operator,
    min_operator,
    protected_div,
)
from skdecide.hub.solver.sgs_policies.sgs_policies import (
    BasePolicyMethod,
    PolicyMethodParams,
)


def fitness_makespan_correlation():
    # domain: RCPSP = load_domain("j301_1.sm")
    domain: RCPSP = load_domain(file_path=get_complete_path("j1201_9.sm"))

    training_domains_names = ["j301_" + str(i) + ".sm" for i in range(1, 11)]
    # training_domains_names =["j1201_9.sm"]

    # evaluation=EvaluationGPHH.PERMUTATION_DISTANCE
    # evaluation = EvaluationGPHH.SGS
    evaluation = EvaluationGPHH.SGS_DEVIATION

    training_domains = []
    for td in training_domains_names:
        training_domains.append(load_domain(file_path=get_complete_path(td)))

    with open("cp_reference_permutations") as json_file:
        cp_reference_permutations = json.load(json_file)

    with open("cp_reference_makespans") as json_file:
        cp_reference_makespans = json.load(json_file)

    set_feature = {
        FeatureEnum.EARLIEST_FINISH_DATE,
        FeatureEnum.EARLIEST_START_DATE,
        FeatureEnum.LATEST_FINISH_DATE,
        FeatureEnum.LATEST_START_DATE,
        FeatureEnum.N_PREDECESSORS,
        FeatureEnum.N_SUCCESSORS,
        FeatureEnum.ALL_DESCENDANTS,
        FeatureEnum.RESSOURCE_REQUIRED,
        FeatureEnum.RESSOURCE_AVG,
        FeatureEnum.RESSOURCE_MAX,
        # FeatureEnum.RESSOURCE_MIN
        FeatureEnum.RESSOURCE_NZ_MIN,
    }

    pset = PrimitiveSet("main", len(set_feature))
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(max_operator, 2)
    pset.addPrimitive(min_operator, 2)
    pset.addPrimitive(operator.neg, 1)

    params_gphh = ParametersGPHH(
        set_feature=set_feature,
        set_primitves=pset,
        tournament_ratio=0.1,
        pop_size=10,
        n_gen=1,
        min_tree_depth=1,
        max_tree_depth=3,
        crossover_rate=0.7,
        mutation_rate=0.3,
        base_policy_method=BasePolicyMethod.SGS_READY,
        delta_index_freedom=0,
        delta_time_freedom=0,
        deap_verbose=True,
        evaluation=evaluation,
        permutation_distance=PermutationDistance.KTD
        # permutation_distance = PermutationDistance.KTD_HAMMING
    )

    solver = GPHH(
        training_domains=training_domains,
        weight=-1,
        verbose=True,
        reference_permutations=cp_reference_permutations,
        # reference_makespans=cp_reference_makespans,
        training_domains_names=training_domains_names,
        params_gphh=params_gphh,
    )

    solver.solve(domain_factory=lambda: domain)

    solver.permutation_distance = PermutationDistance.KTD
    solver.init_reference_permutations(
        cp_reference_permutations, training_domains_names
    )

    random_pop = pop = solver.toolbox.population(n=100)
    print(random_pop)

    out = "f_sgs_train\tf_sgs_dev_train\tf_perm_train\tmk_test\n"
    for ind in random_pop:
        fitness_sgs = solver.evaluate_heuristic(ind, solver.training_domains)[0]
        fitness_sgs_dev = solver.evaluate_heuristic_sgs_deviation(
            ind, solver.training_domains
        )[0]
        fitness_perm = solver.evaluate_heuristic_permutation(
            ind, solver.training_domains
        )[0]
        gphh_policy = GPHHPolicy(
            domain=domain,
            func_heuristic=solver.toolbox.compile(expr=ind),
            features=list(set_feature),
            params_gphh=params_gphh,
        )

        domain.set_inplace_environment(False)
        state = domain.get_initial_state()
        states, actions, values = rollout_episode(
            domain=domain,
            max_steps=1000,
            solver=gphh_policy,
            from_memory=state,
            verbose=False,
            outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
        )

        policy_makespan = states[-1].t

        out += (
            str(fitness_sgs)
            + "\t"
            + str(fitness_sgs_dev)
            + "\t"
            + str(fitness_perm)
            + "\t"
            + str(policy_makespan)
            + "\n"
        )
        print(out)
        print("---------")
    print("DONE")
    print(out)


def run_gphh():

    import time

    n_runs = 1
    makespans = []

    domain: RCPSP = load_domain(file_path=get_complete_path("j601_1.sm"))

    training_domains_names = ["j601_" + str(i) + ".sm" for i in range(1, 11)]

    training_domains = []
    for td in training_domains_names:
        training_domains.append(load_domain(file_path=get_complete_path(td)))

    runtimes = []
    for i in range(n_runs):

        domain.set_inplace_environment(False)
        state = domain.get_initial_state()

        with open("cp_reference_permutations") as json_file:
            cp_reference_permutations = json.load(json_file)

        # with open('cp_reference_makespans') as json_file:
        #     cp_reference_makespans = json.load(json_file)

        start = time.time()

        solver = GPHH(
            training_domains=training_domains,
            domain_model=training_domains[3],
            weight=-1,
            verbose=True,
            reference_permutations=cp_reference_permutations,
            # reference_makespans=cp_reference_makespans,
            training_domains_names=training_domains_names,
            params_gphh=ParametersGPHH.fast_test()
            # params_gphh=ParametersGPHH.default()
        )
        solver.solve(domain_factory=lambda: domain)
        end = time.time()
        runtimes.append((end - start))
        heuristic = solver.hof
        print("ttype:", solver.best_heuristic)
        folder = "./trained_gphh_heuristics"
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = open(os.path.join(folder, "test_gphh_" + str(i) + ".pkl"), "wb")
        pickle.dump(dict(hof=heuristic), file)
        file.close()
        solver.set_domain(domain)
        states, actions, values = rollout_episode(
            domain=domain,
            max_steps=1000,
            solver=solver,
            from_memory=state,
            verbose=False,
            outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
        )
        print("Cost :", sum([v.cost for v in values]))
        makespans.append(sum([v.cost for v in values]))

    print("makespans: ", makespans)
    print("runtimes: ", runtimes)
    print("runtime - mean: ", np.mean(runtimes))


def run_pooled_gphh():

    n_runs = 1
    pool_size = 5
    remove_extreme_values = 1

    makespans = []

    domain: RCPSP = load_domain(file_path=get_complete_path("j1201_9.sm"))
    # domain: RCPSP = load_domain("j1201_9.sm")

    training_domains_names = ["j301_" + str(i) + ".sm" for i in range(1, 11)]

    training_domains = []
    for td in training_domains_names:
        training_domains.append(load_domain(file_path=get_complete_path(td)))

    for i in range(n_runs):

        domain.set_inplace_environment(False)
        state = domain.get_initial_state()

        with open("cp_reference_permutations") as json_file:
            cp_reference_permutations = json.load(json_file)

        heuristics = []
        func_heuristics = []
        folder = "./trained_gphh_heuristics"

        files = os.listdir(folder)
        solver = GPHH(
            training_domains=training_domains,
            domain_model=training_domains[0],
            weight=-1,
            verbose=True,
            reference_permutations=cp_reference_permutations,
            training_domains_names=training_domains_names,
        )

        print("files: ", files)
        for f in files:
            full_path = folder + "/" + f
            print("f: ", full_path)
            tmp = pickle.load(open(full_path, "rb"))
            heuristics.append(tmp)
            func_heuristics.append(solver.toolbox.compile(expr=tmp))

        # for pool in range(pool_size):
        # solver = GPHH(training_domains=training_domains,
        #               weight=-1,
        #               verbose=True,
        #               reference_permutations=cp_reference_permutations,
        #               training_domains_names=training_domains_names
        #               )
        # solver.solve(domain_factory=lambda: domain)
        # func_heuristics.append(solver.func_heuristic)

        pooled_gphh_solver = PooledGPHHPolicy(
            domain=domain,
            domain_model=training_domains[0],
            func_heuristics=func_heuristics,
            features=list(solver.params_gphh.set_feature),
            params_gphh=solver.params_gphh,
            pool_aggregation_method=PoolAggregationMethod.MEAN,
            remove_extremes_values=remove_extreme_values,
        )
        states, actions, values = rollout_episode(
            domain=domain,
            max_steps=1000,
            solver=pooled_gphh_solver,
            from_memory=state,
            verbose=False,
            outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
        )
        print("Cost :", sum([v.cost for v in values]))
        makespans.append(sum([v.cost for v in values]))

    print("makespans: ", makespans)


def run_gphh_with_settings():

    domain: RCPSP = load_domain(get_complete_path("j301_1.sm"))
    training_domains = [
        load_domain(get_complete_path("j301_2.sm")),
        load_domain(get_complete_path("j301_3.sm")),
        load_domain(get_complete_path("j301_4.sm")),
        load_domain(get_complete_path("j301_5.sm")),
        load_domain(get_complete_path("j301_6.sm")),
        load_domain(get_complete_path("j301_7.sm")),
        load_domain(get_complete_path("j301_8.sm")),
        load_domain(get_complete_path("j301_9.sm")),
        load_domain(get_complete_path("j301_10.sm")),
    ]

    domain.set_inplace_environment(False)
    state = domain.get_initial_state()

    set_feature = {
        FeatureEnum.EARLIEST_FINISH_DATE,
        FeatureEnum.EARLIEST_START_DATE,
        FeatureEnum.LATEST_FINISH_DATE,
        FeatureEnum.LATEST_START_DATE,
        FeatureEnum.PRECEDENCE_DONE,
        FeatureEnum.ALL_DESCENDANTS,
        FeatureEnum.RESSOURCE_AVG,
    }

    pset = PrimitiveSet("main", len(set_feature))
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(max_operator, 2)
    pset.addPrimitive(min_operator, 2)
    pset.addPrimitive(operator.neg, 1)

    params_gphh = ParametersGPHH(
        set_feature=set_feature,
        set_primitves=pset,
        tournament_ratio=0.25,
        pop_size=20,
        n_gen=20,
        min_tree_depth=1,
        max_tree_depth=5,
        crossover_rate=0.7,
        mutation_rate=0.1,
        base_policy_method=BasePolicyMethod.SGS_READY,
        delta_index_freedom=0,
        delta_time_freedom=0,
        deap_verbose=True,
    )

    solver = GPHH(
        training_domains=training_domains,
        weight=-1,
        verbose=True,
        params_gphh=params_gphh,
    )
    solver.solve(domain_factory=lambda: domain)
    states, actions, values = rollout_episode(
        domain=domain,
        max_steps=1000,
        solver=solver,
        from_memory=state,
        verbose=False,
        outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
    )
    print("Cost :", sum([v.cost for v in values]))


def compare_settings():
    test_domain_names = ["j1201_1.sm"]
    training_domains_names = ["j301_" + str(i) + ".sm" for i in range(2, 11)]

    domains_loaded = []
    for td in training_domains_names:
        domains_loaded.append(load_domain(get_complete_path(td)))

    n_walks = 5

    all_settings = []

    params1 = ParametersGPHH.default()
    params1.base_policy_method = BasePolicyMethod.SGS_PRECEDENCE
    all_settings.append(params1)

    params2 = ParametersGPHH.default()
    params2.base_policy_method = BasePolicyMethod.SGS_INDEX_FREEDOM
    params2.delta_index_freedom = 5
    all_settings.append(params2)

    params3 = ParametersGPHH.default()
    params3.base_policy_method = BasePolicyMethod.SGS_READY
    all_settings.append(params3)

    params4 = ParametersGPHH.default()
    params4.base_policy_method = BasePolicyMethod.SGS_STRICT
    all_settings.append(params4)

    params5 = ParametersGPHH.default()
    params5.base_policy_method = BasePolicyMethod.SGS_TIME_FREEDOM
    params5.delta_time_freedom = 5
    all_settings.append(params5)

    all_results = {}
    for dom in test_domain_names:
        all_results[dom] = {}
        for par in all_settings:
            print("par: ", par.base_policy_method)
            all_results[dom][par.base_policy_method] = []

    for params in all_settings:
        for i in range(n_walks):
            print("params: ", params.base_policy_method)
            print("walk #", i)
            domain: RCPSP = load_domain(get_complete_path("j301_1.sm"))
            domain.set_inplace_environment(False)
            solver = GPHH(
                training_domains=domains_loaded,
                weight=-1,
                verbose=False,
                params_gphh=params,
            )
            solver.solve(domain_factory=lambda: domain)

            for test_domain_str in test_domain_names:
                domain: RCPSP = load_domain(get_complete_path(test_domain_str))
                domain.set_inplace_environment(False)
                state = domain.get_initial_state()
                solver.set_domain(domain)
                states, actions, values = rollout_episode(
                    domain=domain,
                    max_steps=1000,
                    solver=solver,
                    from_memory=state,
                    verbose=False,
                    outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
                )
                print("One GPHH done")
                print("Best evolved heuristic: ", solver.best_heuristic)
                print("Cost: ", sum([v.cost for v in values]))

                all_results[test_domain_str][params.base_policy_method].append(
                    sum([v.cost for v in values])
                )

    print("##### ALL RESULTS #####")

    for test_domain_str in test_domain_names:
        print(test_domain_str, " :")
        for param_key in all_results[test_domain_str].keys():
            print("\t", param_key, ": ")
            print("\t\t all runs:", all_results[test_domain_str][param_key])
            print("\t\t mean:", np.mean(all_results[test_domain_str][param_key]))


def run_features():
    domain: RCPSP = load_domain(get_complete_path("j301_1.sm"))
    task_id = 2
    total_nres = feature_total_n_res(domain, task_id)
    print("total_nres: ", total_nres)
    duration = feature_task_duration(domain, task_id)
    print("duration: ", duration)
    n_successors = feature_n_successors(domain, task_id)
    print("n_successors: ", n_successors)
    n_predecessors = feature_n_predecessors(domain, task_id)
    print("n_predecessors: ", n_predecessors)
    average_resource_requirements = feature_average_resource_requirements(
        domain, task_id
    )
    print("average_resource_requirements: ", average_resource_requirements)


def run_comparaison_stochastic():
    import random

    from skdecide.hub.domain.rcpsp.rcpsp_sk import (
        RCPSP,
        build_n_determinist_from_stochastic,
        build_stochastic_from_deterministic,
    )

    repeat_runs = 5

    test_domain_names = [
        "j301_1.sm",
        "j301_2.sm",
        "j301_3.sm",
        "j601_1.sm",
        "j601_2.sm",
        "j601_3.sm",
    ]

    all_results = {}
    for dom in test_domain_names:
        all_results[dom] = {
            "random_walk": [],
            "cp": [],
            "cp_sgs": [],
            "gphh": [],
            "pile": [],
        }

    for original_domain_name in test_domain_names:
        original_domain: RCPSP = load_domain(get_complete_path(original_domain_name))
        task_to_noise = set(
            random.sample(
                original_domain.get_tasks_ids(), len(original_domain.get_tasks_ids())
            )
        )
        stochastic_domain = build_stochastic_from_deterministic(
            original_domain, task_to_noise=task_to_noise
        )
        deterministic_domains = build_n_determinist_from_stochastic(
            stochastic_domain, nb_instance=6
        )

        training_domains = deterministic_domains[0:-1]
        training_domains_names = [None for i in range(len(training_domains))]
        test_domain = deterministic_domains[-1]
        print("training_domains:", training_domains)

        # RANDOM WALK
        domain: RCPSP = test_domain
        domain.set_inplace_environment(False)
        # random_walk_costs = []
        for i in range(repeat_runs):
            state = domain.get_initial_state()
            solver = None
            states, actions, values = rollout_episode(
                domain=domain,
                max_steps=1000,
                solver=solver,
                from_memory=state,
                verbose=False,
                outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
            )
            print("One random Walk complete")
            print("Cost :", sum([v.cost for v in values]))
            all_results[original_domain_name]["random_walk"].append(
                sum([v.cost for v in values])
            )
        print("All random Walk complete")

        # CP
        domain = test_domain
        do_solver = SolvingMethod.CP
        domain.set_inplace_environment(False)
        state = domain.get_initial_state()
        solver = DOSolver(
            policy_method_params=PolicyMethodParams(
                base_policy_method=BasePolicyMethod.FOLLOW_GANTT,
                delta_index_freedom=0,
                delta_time_freedom=0,
            ),
            method=do_solver,
        )
        solver.solve(domain_factory=lambda: domain)
        print(do_solver)
        states, actions, values = rollout_episode(
            domain=domain,
            solver=solver,
            from_memory=state,
            max_steps=500,
            verbose=False,
            outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
        )
        print("Cost: ", sum([v.cost for v in values]))
        print("CP done")
        all_results[original_domain_name]["cp"].append(sum([v.cost for v in values]))

        # CP SGS
        for train_dom in training_domains:
            do_solver = SolvingMethod.CP
            train_dom.set_inplace_environment(False)
            state = train_dom.get_initial_state()
            solver = DOSolver(
                policy_method_params=PolicyMethodParams(
                    base_policy_method=BasePolicyMethod.SGS_STRICT,
                    delta_index_freedom=0,
                    delta_time_freedom=0,
                ),
                method=do_solver,
            )
            solver.solve(domain_factory=lambda: train_dom)
            print(do_solver)
            domain: RCPSP = test_domain
            domain.set_inplace_environment(False)
            states, actions, values = rollout_episode(
                domain=domain,
                solver=solver,
                from_memory=state,
                max_steps=500,
                verbose=False,
                outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
            )
            print("Cost: ", sum([v.cost for v in values]))
            print("CP_SGS done")
            all_results[original_domain_name]["cp_sgs"].append(
                sum([v.cost for v in values])
            )

        # PILE
        domain: RCPSP = test_domain
        do_solver = SolvingMethod.PILE
        domain.set_inplace_environment(False)
        state = domain.get_initial_state()
        solver = DOSolver(
            policy_method_params=PolicyMethodParams(
                base_policy_method=BasePolicyMethod.FOLLOW_GANTT,
                delta_index_freedom=0,
                delta_time_freedom=0,
            ),
            method=do_solver,
        )
        solver.solve(domain_factory=lambda: domain)
        print(do_solver)
        states, actions, values = rollout_episode(
            domain=domain,
            solver=solver,
            from_memory=state,
            max_steps=500,
            verbose=False,
            outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
        )
        print("Cost: ", sum([v.cost for v in values]))
        print("PILE done")
        all_results[original_domain_name]["pile"].append(sum([v.cost for v in values]))

        # GPHH
        with open("cp_reference_permutations") as json_file:
            cp_reference_permutations = json.load(json_file)

        with open("cp_reference_makespans") as json_file:
            cp_reference_makespans = json.load(json_file)

        for i in range(repeat_runs):
            domain.set_inplace_environment(False)

            set_feature = {
                FeatureEnum.EARLIEST_FINISH_DATE,
                FeatureEnum.EARLIEST_START_DATE,
                FeatureEnum.LATEST_FINISH_DATE,
                FeatureEnum.LATEST_START_DATE,
                FeatureEnum.N_PREDECESSORS,
                FeatureEnum.N_SUCCESSORS,
                FeatureEnum.ALL_DESCENDANTS,
                FeatureEnum.RESSOURCE_REQUIRED,
                FeatureEnum.RESSOURCE_AVG,
                FeatureEnum.RESSOURCE_MAX,
                # FeatureEnum.RESSOURCE_MIN
                FeatureEnum.RESSOURCE_NZ_MIN,
            }

            pset = PrimitiveSet("main", len(set_feature))
            pset.addPrimitive(operator.add, 2)
            pset.addPrimitive(operator.sub, 2)
            pset.addPrimitive(operator.mul, 2)
            pset.addPrimitive(protected_div, 2)
            pset.addPrimitive(max_operator, 2)
            pset.addPrimitive(min_operator, 2)
            pset.addPrimitive(operator.neg, 1)
            # pset.addPrimitive(operator.pow, 2)

            params_gphh = ParametersGPHH(
                set_feature=set_feature,
                set_primitves=pset,
                tournament_ratio=0.2,
                pop_size=40,
                n_gen=20,
                min_tree_depth=1,
                max_tree_depth=3,
                crossover_rate=0.7,
                mutation_rate=0.3,
                base_policy_method=BasePolicyMethod.SGS_READY,
                delta_index_freedom=0,
                delta_time_freedom=0,
                deap_verbose=True,
                evaluation=EvaluationGPHH.SGS_DEVIATION,
                permutation_distance=PermutationDistance.KTD
                # permutation_distance = PermutationDistance.KTD_HAMMING
            )

            solver = GPHH(
                training_domains=training_domains,
                weight=-1,
                verbose=False,
                reference_permutations=cp_reference_permutations,
                # reference_makespans=cp_reference_makespans,
                training_domains_names=training_domains_names,
                params_gphh=params_gphh
                # set_feature=set_feature)
            )
            solver.solve(domain_factory=lambda: domain)

            domain: RCPSP = test_domain
            domain.set_inplace_environment(False)
            state = domain.get_initial_state()
            solver.set_domain(domain)
            states, actions, values = rollout_episode(
                domain=domain,
                max_steps=1000,
                solver=solver,
                from_memory=state,
                verbose=False,
                outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
            )
            print("One GPHH done")
            print("Best evolved heuristic: ", solver.best_heuristic)
            print("Cost: ", sum([v.cost for v in values]))

            all_results[original_domain_name]["gphh"].append(
                sum([v.cost for v in values])
            )

    print("##### ALL RESULTS #####")

    for test_domain_str in test_domain_names:
        print(test_domain_str, " :")
        for algo_key in all_results[test_domain_str].keys():
            print("\t", algo_key, ": ")
            print("\t\t all runs:", all_results[test_domain_str][algo_key])
            print("\t\t mean:", np.mean(all_results[test_domain_str][algo_key]))


def run_comparaison():
    import os

    from examples.scheduling.rcpsp_datasets import get_data_available

    files = get_data_available()
    all_single_mode = [os.path.basename(f) for f in files if "sm" in f]
    # training_cphh = ["j1201_"+str(i)+".sm" for i in range(2, 11)]
    training_cphh = ["j301_" + str(i) + ".sm" for i in range(1, 11)]

    # all_testing_domains_names = [f for f in all_single_mode
    #                              if not(any(g in f for g in training_cphh))]
    all_testing_domains_names = ["j1201_2.sm"]
    # all_testing_domains_names = ["j601_2.sm"]

    # all_testing_domains_names = random.sample(all_testing_domains_names, 1)
    # training_domains_names = [f for f in all_single_mode
    #                           if any(g in f for g in training_cphh)]

    training_domains_names = all_testing_domains_names
    domains_loaded = {
        domain_name: load_domain(get_complete_path(domain_name))
        for domain_name in all_testing_domains_names
    }
    test_domain_names = all_testing_domains_names
    # test_domain_names = [test_domain_names[-1]]
    # test_domain_names = ["j1201_1.sm"]
    print("test_domain_names: ", test_domain_names)
    print("training_domains_names: ", training_domains_names)
    n_walks = 5
    for td in training_domains_names:
        domains_loaded[td] = load_domain(get_complete_path(td))

    all_results = {}
    for dom in test_domain_names:
        all_results[dom] = {
            "random_walk": [],
            "cp": [],
            "cp_sgs": [],
            "gphh": [],
            "pile": [],
        }

    # RANDOM WALK
    for test_domain_str in test_domain_names:
        domain: RCPSP = domains_loaded[test_domain_str]
        domain.set_inplace_environment(False)
        n_walks = 5
        for i in range(n_walks):
            state = domain.get_initial_state()
            solver = None
            states, actions, values = rollout_episode(
                domain=domain,
                max_steps=1000,
                solver=solver,
                from_memory=state,
                verbose=False,
                outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
            )
            print("One random Walk complete")
            print("Cost :", sum([v.cost for v in values]))
            all_results[test_domain_str]["random_walk"].append(
                sum([v.cost for v in values])
            )
        print("All random Walk complete")

    # CP
    for test_domain_str in test_domain_names:
        domain: RCPSP = domains_loaded[test_domain_str]
        do_solver = SolvingMethod.CP
        domain.set_inplace_environment(False)
        state = domain.get_initial_state()
        solver = DOSolver(
            policy_method_params=PolicyMethodParams(
                base_policy_method=BasePolicyMethod.FOLLOW_GANTT,
                delta_index_freedom=0,
                delta_time_freedom=0,
            ),
            method=do_solver,
        )
        solver.solve(domain_factory=lambda: domain)
        print(do_solver)
        states, actions, values = rollout_episode(
            domain=domain,
            solver=solver,
            from_memory=state,
            max_steps=500,
            verbose=False,
            outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
        )
        print("Cost: ", sum([v.cost for v in values]))
        print("CP done")
        all_results[test_domain_str]["cp"].append(sum([v.cost for v in values]))

    # CP SGS
    for test_domain_str in test_domain_names:
        domain: RCPSP = domains_loaded[test_domain_str]
        do_solver = SolvingMethod.CP
        domain.set_inplace_environment(False)
        state = domain.get_initial_state()
        solver = DOSolver(
            policy_method_params=PolicyMethodParams(
                base_policy_method=BasePolicyMethod.SGS_STRICT,
                delta_index_freedom=0,
                delta_time_freedom=0,
            ),
            method=do_solver,
        )
        solver.solve(domain_factory=lambda: domain)
        print(do_solver)
        states, actions, values = rollout_episode(
            domain=domain,
            solver=solver,
            from_memory=state,
            max_steps=500,
            verbose=False,
            outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
        )
        print("Cost: ", sum([v.cost for v in values]))
        print("CP_SGS done")
        all_results[test_domain_str]["cp_sgs"].append(sum([v.cost for v in values]))

    # PILE
    for test_domain_str in test_domain_names:
        domain: RCPSP = domains_loaded[test_domain_str]
        do_solver = SolvingMethod.PILE
        domain.set_inplace_environment(False)
        state = domain.get_initial_state()
        solver = DOSolver(
            policy_method_params=PolicyMethodParams(
                base_policy_method=BasePolicyMethod.FOLLOW_GANTT,
                delta_index_freedom=0,
                delta_time_freedom=0,
            ),
            method=do_solver,
        )
        solver.solve(domain_factory=lambda: domain)
        print(do_solver)
        states, actions, values = rollout_episode(
            domain=domain,
            solver=solver,
            from_memory=state,
            max_steps=500,
            verbose=False,
            outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
        )
        print("Cost: ", sum([v.cost for v in values]))
        print("PILE done")
        all_results[test_domain_str]["pile"].append(sum([v.cost for v in values]))

    # GPHH
    domain: RCPSP = load_domain(get_complete_path("j301_1.sm"))
    training_domains = [
        domains_loaded[training_domain] for training_domain in training_domains_names
    ]

    with open("cp_reference_permutations") as json_file:
        cp_reference_permutations = json.load(json_file)

    # with open('cp_reference_makespans') as json_file:
    #     cp_reference_makespans = json.load(json_file)

    for i in range(n_walks):
        domain.set_inplace_environment(False)

        set_feature = {
            FeatureEnum.EARLIEST_FINISH_DATE,
            FeatureEnum.EARLIEST_START_DATE,
            FeatureEnum.LATEST_FINISH_DATE,
            FeatureEnum.LATEST_START_DATE,
            FeatureEnum.N_PREDECESSORS,
            FeatureEnum.N_SUCCESSORS,
            FeatureEnum.ALL_DESCENDANTS,
            FeatureEnum.RESSOURCE_REQUIRED,
            FeatureEnum.RESSOURCE_AVG,
            FeatureEnum.RESSOURCE_MAX,
            # FeatureEnum.RESSOURCE_MIN
            FeatureEnum.RESSOURCE_NZ_MIN,
        }

        pset = PrimitiveSet("main", len(set_feature))
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protected_div, 2)
        pset.addPrimitive(max_operator, 2)
        pset.addPrimitive(min_operator, 2)
        pset.addPrimitive(operator.neg, 1)
        # pset.addPrimitive(operator.pow, 2)

        params_gphh = ParametersGPHH(
            set_feature=set_feature,
            set_primitves=pset,
            tournament_ratio=0.2,
            pop_size=20,
            n_gen=7,
            min_tree_depth=1,
            max_tree_depth=3,
            crossover_rate=0.7,
            mutation_rate=0.3,
            base_policy_method=BasePolicyMethod.SGS_READY,
            delta_index_freedom=0,
            delta_time_freedom=0,
            deap_verbose=True,
            evaluation=EvaluationGPHH.SGS_DEVIATION,
            permutation_distance=PermutationDistance.KTD
            # permutation_distance = PermutationDistance.KTD_HAMMING
        )

        solver = GPHH(
            training_domains=training_domains,
            weight=-1,
            verbose=False,
            reference_permutations=cp_reference_permutations,
            # reference_makespans=cp_reference_makespans,
            training_domains_names=training_domains_names,
            params_gphh=params_gphh,
        )
        solver.solve(domain_factory=lambda: domain)

        for test_domain_str in test_domain_names:
            domain: RCPSP = domains_loaded[test_domain_str]
            domain.set_inplace_environment(False)
            state = domain.get_initial_state()
            solver.set_domain(domain)
            states, actions, values = rollout_episode(
                domain=domain,
                max_steps=1000,
                solver=solver,
                from_memory=state,
                verbose=False,
                outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
            )
            print("One GPHH done")
            print("Best evolved heuristic: ", solver.best_heuristic)
            print("Cost: ", sum([v.cost for v in values]))

            all_results[test_domain_str]["gphh"].append(sum([v.cost for v in values]))

    print("All GPHH done")

    print("##### ALL RESULTS #####")

    for test_domain_str in test_domain_names:
        print(test_domain_str, " :")
        for algo_key in all_results[test_domain_str].keys():
            print("\t", algo_key, ": ")
            print("\t\t all runs:", all_results[test_domain_str][algo_key])
            print("\t\t mean:", np.mean(all_results[test_domain_str][algo_key]))

    # print('random walks: ', random_walk_costs)
    # print('random walk (mean): ', np.mean(random_walk_costs))
    # print('CP: ', cp_cost)
    # print('GPHH: ', gphh_costs)
    # print('GPHH (mean): ', np.mean(gphh_costs))
    # import json
    # import time
    # from datetime import datetime
    # all_results["params"] = {"features": [str(k.value) for k in set_feature]}
    # json.dump(all_results, open("training_on_big_benchmark_gphh_cp_random_"+str(datetime.now()), "w"), indent=2)


def compute_ref_permutations():
    import os

    files = get_data_available()
    all_single_mode = [os.path.basename(f) for f in files if "sm" in f]

    all_permutations = {}
    all_makespans = {}
    for td_name in all_single_mode:
        td = load_domain(get_complete_path(td_name))
        td.set_inplace_environment(False)
        solver = DOSolver(
            policy_method_params=PolicyMethodParams(
                base_policy_method=BasePolicyMethod.SGS_PRECEDENCE,
                delta_index_freedom=0,
                delta_time_freedom=0,
            ),
            method=SolvingMethod.CP,
        )
        solver.solve(domain_factory=lambda: td)
        raw_permutation = solver.best_solution.rcpsp_permutation
        full_permutation = [int(x + 2) for x in raw_permutation]
        full_permutation.insert(0, 1)
        full_permutation.append(int(np.max(full_permutation) + 1))
        print("full_perm: ", full_permutation)
        all_permutations[td_name] = full_permutation

        state = td.get_initial_state()
        states, actions, values = rollout_episode(
            domain=td,
            max_steps=1000,
            solver=solver,
            from_memory=state,
            verbose=False,
            outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
        )

        makespan = sum([v.cost for v in values])
        all_makespans[td_name] = makespan
        print("makespan: ", makespan)

    print("all_permutations: ", all_permutations)
    print("all_makespans: ", all_makespans)

    json.dump(all_permutations, open("cp_reference_permutations", "w"), indent=2)
    json.dump(all_makespans, open("cp_reference_makespans", "w"), indent=2)


if __name__ == "__main__":
    run_gphh()
