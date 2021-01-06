from examples.scheduling.toy_rcpsp_examples import MyExampleRCPSPDomain, MyExampleMRCPSPDomain_WithCost
from skdecide.hub.solver.do_solver.sk_to_do_binding import build_do_domain


def create_do_from_sk():
    rcpsp_domain = MyExampleRCPSPDomain()
    do_domain = build_do_domain(rcpsp_domain)
    print(do_domain.__class__)
    rcpsp_domain = MyExampleMRCPSPDomain_WithCost()
    do_domain = build_do_domain(rcpsp_domain)
    print(do_domain.__class__)
    from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_domain, load_multiskill_domain
    rcpsp_domain = load_multiskill_domain()
    do_domain = build_do_domain(rcpsp_domain)
    print(do_domain.__class__)


if __name__ == "__main__":
    create_do_from_sk()
