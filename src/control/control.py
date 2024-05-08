import numpy as np


class Control():
    """
    Base control class
    """
    control_type = "Base"

    def __init__(self, dof: int = 2) -> None:
        self.dof = dof

    def step(self, x_init, u_init, x_desired) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Steps controller

        Parameters 
        ----------
        x_init : np.ndarray
            Initial state vector, i.e. [x, y, theta, u, v, r]
        u_init : np.ndarray
            Initial control signal
        x_desired : np.ndarray
            Goal state vector

        Returns
        -------
        x : np.ndarray
            State solution within the given horizon
        u : np.ndarray
            Optimal control vector within the given horizon
        """
        pass

    def debug(self, x_init: np.ndarray, u_init: np.ndarray,
              x_desired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Steps controller with a try-except block and gives a crash report 
        if something goes wrong.

        Parameters
        ----------
        x_init : np.ndarray
            Initial state vector, i.e. [x, y, theta, u, v, r]
        u_init : np.ndarray
            Initial control signal
        x_desired : np.ndarray
            Goal state vector

        Returns
        -------
        x : np.ndarray
            State solution within the given horizon
        u : np.ndarray
            Optimal control vector within the given horizon
        """

        ...


class Manual(Control):
    """
    Manual vehicle control
    """
    control_type = "Manual"

    def __init__(self, dof: int = 2) -> None:
        super().__init__(dof)
