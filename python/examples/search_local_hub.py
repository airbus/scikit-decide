from airlaps import hub, Domain, Solver

if __name__ == '__main__':

    print('Downloaded hub domains:', hub.local_search(Domain))
    print('Downloaded hub solvers:', hub.local_search(Solver))
