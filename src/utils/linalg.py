import numpy as np


# ------------------------------------------------------------------------------


def moore_penrose(A: np.ndarray):
    """
    Right hand Moore-Penrose pseudo-inverse

    A^T(AA^T)^-1

    Parameters
    ----------
        A : np.ndarray
            Non-invertible matrix

    Returns
    -------
        moore_penrose : np.ndarray
            Inverted matrix
    """

    return A.T.dot(np.linalg.inv(A.dot(A.T)))
