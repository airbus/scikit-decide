from skdecide.builders.discrete_optimization.rcpsp.solver.ls_solver import LS_RCPSP_Solver, LS_SOLVER
from skdecide.builders.discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import CP_MS_MRCPSP_MZN
from skdecide.builders.discrete_optimization.rcpsp_multiskill.solvers.lp_model import LP_Solver_MRSCPSP

from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage

from skdecide.builders.discrete_optimization.rcpsp.solver.cp_solvers import CP_RCPSP_MZN, CP_MRCPSP_MZN, CPSolverName

from skdecide.builders.discrete_optimization.generic_tools.lp_tools import ParametersMilp
from skdecide.builders.discrete_optimization.generic_tools.cp_tools import ParametersCP
from skdecide.builders.discrete_optimization.rcpsp_multiskill.solvers.lp_model import LP_Solver_MRSCPSP, MilpSolverName
from skdecide.builders.discrete_optimization.rcpsp_multiskill.solvers.ms_rcpsp_cp_lns_solver import LNS_CP_MS_RCPSP_SOLVER, OptionNeighbor
from skdecide.builders.discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPModel, MS_RCPSPModel_Variant
from skdecide.builders.discrete_optimization.generic_tools.ea.ga_tools import ParametersAltGa
from skdecide.builders.discrete_optimization.rcpsp_multiskill.solvers.ms_rcpsp_ga_solver import GA_MSRCPSP_Solver
from skdecide.builders.discrete_optimization.rcpsp_multiskill.solvers.calendar_solver_iterative import SolverWithCalendarIterative

solvers = {"lp": [(LP_Solver_MRSCPSP, {"lp_solver": MilpSolverName.CBC, "parameters_milp": ParametersMilp.default()})],
           "cp": [(CP_MS_MRCPSP_MZN, {"cp_solver_name": CPSolverName.CHUFFED,
                                      "parameters_cp": ParametersCP.default()})],
           "lns": [(LNS_CP_MS_RCPSP_SOLVER, {"nb_iteration_lns": 500,
                                             "option_neighbor": OptionNeighbor.MIX_LARGE_NEIGH})],
           "lns-cp": [(LNS_CP_MS_RCPSP_SOLVER, {"nb_iteration_lns": 20,
                                                "option_neighbor": OptionNeighbor.MIX_LARGE_NEIGH})],
           "ls": [(LS_RCPSP_Solver, {"ls_solver": LS_SOLVER.SA,
                                     "nb_iteration_max": 20})],
           "ga": [(GA_MSRCPSP_Solver, {"parameters_ga": ParametersAltGa.default_msrcpsp()})],
           "lns-cp-calendar": [(SolverWithCalendarIterative, {"option_neighbor": OptionNeighbor.MIX_LARGE_NEIGH,
                                                              "parameters_cp": ParametersCP.default(),
                                                              "nb_iteration_lns": 20,
                                                              "skip_first_iteration": False})]
           }

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility = {LP_Solver_MRSCPSP: [MS_RCPSPModel, MS_RCPSPModel_Variant],
                         SolverWithCalendarIterative: [MS_RCPSPModel, MS_RCPSPModel_Variant],
                         CP_MS_MRCPSP_MZN: [MS_RCPSPModel, MS_RCPSPModel_Variant],
                         LNS_CP_MS_RCPSP_SOLVER: [MS_RCPSPModel, MS_RCPSPModel_Variant],
                         LS_RCPSP_Solver: [MS_RCPSPModel, MS_RCPSPModel_Variant],
                         GA_MSRCPSP_Solver: [MS_RCPSPModel_Variant]}


def look_for_solver(domain):
    class_domain = domain.__class__
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    print("You have ", len(available), " solvers for your domain ")
    print([solvers_map[a] for a in available])
    return available


def solve(method,
          rcpsp_model: MS_RCPSPModel,
          **args)->ResultStorage:
    solver = method(rcpsp_model, **args)
    try:
        solver.init_model(**args)
    except:
        pass
    return solver.solve(**args)


def return_solver(method,
                  rcpsp_model: MS_RCPSPModel,
                  **args)->ResultStorage:
    solver = method(rcpsp_model, **args)
    try:
        solver.init_model(**args)
    except:
        pass
    return solver
