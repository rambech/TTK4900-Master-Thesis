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


# ------------------------------------------------------------------------------

def singular_projection(A, sing_val_thres=1e-2, svd_tol=1e-7):
    """
    Makes a singular projection based on the smallest value below 
    sing_val_thres and returns the projection. If a projection could 
    not be made, an identity matrix of size A.shape[0] will be returned

    Additionally, we check if the SVD is within svd_tol

    Parameters
    ----------
        A : np.ndarray
            Any matrix
        sing_val_thres : float
            Upper limit to where we can pick singular values from
        svd_tol : float
            Upper limit on absolute SVD recomposition error

    Returns
    -------
        projection : np.ndarray
            Singular projection or identity matrix if a singular 
            projection could not be made
    """

    U, S, Vh = np.linalg.svd(A)

    A_recomposed = U @ np.diag(S) @ Vh
    print(f"difference: {np.abs(A - A_recomposed)}")

    if np.allclose(A, A_recomposed, atol=svd_tol):
        mask = S < sing_val_thres
        if np.any(mask):
            Vh_reduced = Vh[mask]
            return Vh_reduced.T @ Vh_reduced
        else:
            print(f"No singular values, projection not made")
    else:
        print(f"Recomposition of SVD matrices failed, projection not made")

    return np.eye(A.shape[0])
