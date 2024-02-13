import numpy as np
from vehicle import DubinsCar
from vehicle.models import DubinsCarModel
from control import NMPC
from maps import SimpleMap, Target
from simulator import Simulator
from control.tests import new_distance_example, test_mpc

# new_distance_example()
# test_mpc()


def test_mpc_simulator():
    """
    Procedure for testing simulator
    """
    # Initialize constants
    control_fps = 20
    sim_fps = 60
    N = 40
    eta_init = np.array([0, 0, 0, 0, 0, 0], float)
    eta_d = np.array([25/2-0.75-1, 0, 0, 0, 0, 0], float)

    # Initialize vehicle
    vehicle = DubinsCar(dt=1/sim_fps)
    model = DubinsCarModel(dt=1/control_fps)
    control = NMPC(model=model, horizon=N, dt=1/control_fps)

    map = SimpleMap()
    target = Target(eta_d, vehicle, map.origin)
    simulator = Simulator(vehicle, control, map, None, target,
                          eta_init=eta_init, fps=control_fps)
    simulator.simulate()


test_mpc_simulator()
