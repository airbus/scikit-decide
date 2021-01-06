from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage

from skdecide.builders.discrete_optimization.rcpsp.solver.cp_solvers import CP_RCPSP_MZN, CP_MRCPSP_MZN, CPSolverName
from skdecide.builders.discrete_optimization.rcpsp.solver.rcpsp_lp_solver import LP_RCPSP, LP_MRCPSP, LP_RCPSP_Solver
from skdecide.builders.discrete_optimization.rcpsp.solver.rcpsp_pile import PileSolverRCPSP, PileSolverRCPSP_Calendar, GreedyChoice
from skdecide.builders.discrete_optimization.rcpsp.solver.rcpsp_lp_lns_solver import InitialSolutionRCPSP, ConstraintHandlerStartTimeInterval, \
    ConstraintHandlerStartTimeIntervalMRCPSP, ConstraintHandlerFixStartTime, LNS_LP_RCPSP_SOLVER
from skdecide.builders.discrete_optimization.rcpsp.solver.rcpsp_cp_lns_solver import ConstraintHandlerStartTimeInterval_CP,\
    LNS_CP_RCPSP_SOLVER
from skdecide.builders.discrete_optimization.rcpsp.solver.ls_solver import LS_SOLVER, LS_RCPSP_Solver
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, MultiModeRCPSPModel, \
    RCPSPModelCalendar, RCPSP_H_Model, SingleModeRCPSPModel
from skdecide.builders.discrete_optimization.rcpsp.solver.calendar_solver_iterative import SolverWithCalendarIterative
from skdecide.builders.discrete_optimization.generic_tools.lp_tools import ParametersMilp
from skdecide.builders.discrete_optimization.generic_tools.cp_tools import ParametersCP
from skdecide.builders.discrete_optimization.generic_tools.ea.ga_tools import ParametersGa, ParametersAltGa
from skdecide.builders.discrete_optimization.rcpsp.solver.rcpsp_ga_solver import GA_RCPSP_Solver, GA_MRCPSP_Solver


solvers = {"lp": [(LP_RCPSP, {"lp_solver": LP_RCPSP_Solver.CBC, "parameters_milp": ParametersMilp.default()}),
                  (LP_MRCPSP, {"lp_solver": LP_RCPSP_Solver.CBC, "parameters_milp": ParametersMilp.default()})],
           "greedy": [(PileSolverRCPSP,  {"greedy_choice": GreedyChoice.MOST_SUCCESSORS}),
                      (PileSolverRCPSP_Calendar, {"greedy_choice": GreedyChoice.MOST_SUCCESSORS})],
           "cp": [(CP_RCPSP_MZN, {"cp_solver_name": CPSolverName.CHUFFED,
                                  "parameters_cp": ParametersCP.default()}),
                  (CP_MRCPSP_MZN, {"cp_solver_name": CPSolverName.CHUFFED,
                                   "parameters_cp": ParametersCP.default()})],
           "lns": [(LNS_LP_RCPSP_SOLVER, {"nb_iteration_lns": 500, "lp_solver": LP_RCPSP_Solver.GRB}),
                   (LNS_CP_RCPSP_SOLVER, {"nb_iteration_lns": 500})],
           "lns-lp": [(LNS_LP_RCPSP_SOLVER, {"nb_iteration_lns": 500, "lp_solver": LP_RCPSP_Solver.GRB})],
           "lns-cp": [(LNS_CP_RCPSP_SOLVER, {"nb_iteration_lns": 500})],
           "ls": [(LS_RCPSP_Solver, {"ls_solver": LS_SOLVER.SA, "nb_iteration_max": 2000})],
           "ga": [(GA_RCPSP_Solver, {"parameters_ga": ParametersGa.default_rcpsp()}),
                  (GA_MRCPSP_Solver, {"parameters_ga": ParametersAltGa.default_mrcpsp()})],
           "lns-cp-calendar": [(SolverWithCalendarIterative, {"parameters_cp": ParametersCP.default(),
                                                              "nb_iteration_lns": 20,
                                                              "skip_first_iteration": False})]
           }

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility = {LP_RCPSP: [SingleModeRCPSPModel],
                         LP_MRCPSP: [MultiModeRCPSPModel,
                                     SingleModeRCPSPModel,
                                     RCPSPModelCalendar],
                         PileSolverRCPSP: [SingleModeRCPSPModel,
                                           MultiModeRCPSPModel],
                         PileSolverRCPSP_Calendar: [SingleModeRCPSPModel,
                                                    MultiModeRCPSPModel,
                                                    RCPSPModelCalendar],
                         CP_RCPSP_MZN: [SingleModeRCPSPModel],
                         CP_MRCPSP_MZN: [SingleModeRCPSPModel,
                                         MultiModeRCPSPModel,
                                         RCPSPModelCalendar],
                         LNS_LP_RCPSP_SOLVER: [SingleModeRCPSPModel,
                                               MultiModeRCPSPModel,
                                               RCPSPModelCalendar],
                         LNS_CP_RCPSP_SOLVER: [SingleModeRCPSPModel,
                                               MultiModeRCPSPModel,
                                               RCPSPModelCalendar],
                         LS_RCPSP_Solver: [SingleModeRCPSPModel,
                                           MultiModeRCPSPModel,
                                           RCPSPModelCalendar],
                         GA_RCPSP_Solver: [SingleModeRCPSPModel],
                         GA_MRCPSP_Solver: [MultiModeRCPSPModel],
                         SolverWithCalendarIterative: [RCPSPModelCalendar,
                                                       SingleModeRCPSPModel,
                                                       MultiModeRCPSPModel]
                         }


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
          rcpsp_model: RCPSPModel,
          **args)->ResultStorage:
    solver = method(rcpsp_model, **args)
    try:
        solver.init_model(**args)
    except:
        pass
    return solver.solve(**args)


def return_solver(method,
                  rcpsp_model: RCPSPModel,
                  **args)->ResultStorage:
    solver = method(rcpsp_model, **args)
    try:
        solver.init_model(**args)
    except:
        pass
    return solver
