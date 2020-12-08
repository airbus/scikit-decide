def generic_mutate_wrapper(individual, problem, encoding_name, indpb, solution_fn, custom_mutation):
    kwargs = {encoding_name: individual, 'problem': problem}
    custom_sol = solution_fn(**kwargs)
    new_custom_sol = custom_mutation.mutate(custom_sol)[0]
    new_individual = individual  # TODO: check if that is correct (only way to keep the individual type I found so far)
    tmp_vector = getattr(new_custom_sol, encoding_name)
    for i in range(len(tmp_vector)):
        new_individual[i] = tmp_vector[i]
    return new_individual,

