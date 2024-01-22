import numpy as np
from vehicle.models import Model


class nmpc():
    def __init__(self, model: Model, horizon: int = 40) -> None:
        self.N = horizon  # Optimization horizon

    def step(self, eta, nu) -> np.ndarray:
        """
        Steps nmpc controller

        Parameters 
        ----------
        eta : np.ndarray
            Current pose vector in 3 DOF
        nu : np.ndarray
            Current velocity vector in 3 DOF

        Returns
        -------
        u : np.ndarray
            Optimal control vector within the given horizon
        """
        pass
