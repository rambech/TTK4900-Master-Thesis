class Control():
    """
    Base control class
    """
    control_type = "Base"

    def __init__(self, dof: int = 2) -> None:
        self.dof = dof

    def step():
        pass


class Manual(Control):
    """
    Manual vehicle control
    """
    control_type = "Manual"

    def __init__(self, dof: int = 2) -> None:
        super().__init__(dof)
