from skdecide.builders.discrete_optimization.facility.facility_model import FacilityProblem, Facility, Point, \
    Customer, FacilityProblem2DPoints
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/facility/")
files_available = [os.path.join(path_to_data, f) for f in os.listdir(path_to_data)]


def parse(input_data):
    # parse the input
    lines = input_data.split('\n')
    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    facilities = []
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.append(Facility(i - 1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3]))))
    customers = []
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.append(Customer(i - 1 - facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))
    problem = FacilityProblem2DPoints(facility_count, customer_count, facilities, customers)
    return problem


def parse_file(file_path)->FacilityProblem:
    with open(file_path, 'r') as input_data_file:
        input_data = input_data_file.read()
        facility_model = parse(input_data)
        return facility_model
