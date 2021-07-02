from examples.scheduling.toy_rcpsp_examples import MyExampleRCPSPDomain, MyExampleMRCPSPDomain_WithCost
from skdecide.hub.solver.do_solver.sk_to_do_binding import build_do_domain

# Testing the binding between skdecide and discrete-optimization lib
def create_do_from_sk():
    rcpsp_domain = MyExampleRCPSPDomain()
    do_domain = build_do_domain(rcpsp_domain)
    print("Loading rcpsp domain :resulting class in DO : ", do_domain.__class__)
    rcpsp_domain = MyExampleMRCPSPDomain_WithCost()
    do_domain = build_do_domain(rcpsp_domain)
    print("Loading multimode-rcpsp domain : resulting class in DO : ", do_domain.__class__)
    from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_multiskill_domain
    from examples.discrete_optimization.rcpsp_multiskill_parser_example import get_data_available_ms
    rcpsp_domain = load_multiskill_domain(get_data_available_ms()[0])
    do_domain = build_do_domain(rcpsp_domain)
    print("Loading multiskill-rcpsp domain : resulting class in DO : ", do_domain.__class__)


if __name__ == "__main__":
    create_do_from_sk()
