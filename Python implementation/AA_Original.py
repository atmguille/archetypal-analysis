"""
Implementation of the original algorithm for the archetypal analysis.

Author: Guillermo García Cobo
"""

import scipy.optimize
import numpy as np

from AA_Abstract import AA_Abstract


class AA_Original(AA_Abstract):

    def __init__(self, n_archetypes: int, max_iter: int = 100, tol: float = 1e-6, verbose: bool = False):
        super().__init__(n_archetypes, max_iter, tol, verbose)

    def _computeA(self, X: np.ndarray, Z: np.ndarray, A: np.ndarray = None) -> np.ndarray:
        return self.__optimize_weights(u=X, T=Z)

    def _computeB(self, X: np.ndarray, A: np.ndarray, B: np.ndarray = None) -> np.ndarray:
        Z = self._computeZ_fromA(X, A)
        return self.__optimize_weights(u=Z, T=X)

    @staticmethod
    def _computeZ_fromA(X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Solve the optimization problem:
            argmin_Z ||X - AZ||_2
        :param X: data matrix, with shape (n_samples, n_features)
        :param A: A matrix, with shape (n_samples, n_archetypes)
        :return: archetypes matrix, with shape (n_archetypes, n_features)
        """
        return np.linalg.lstsq(a=A, b=X, rcond=None)[0]

    @staticmethod
    def __optimize_weights(u: np.ndarray, T: np.ndarray, huge_constant: int = 200) -> np.ndarray:
        """
        Solve the optimization problem:
            argmin_w ||u - wT||_2 + huge_constant * ||1 - w||_2
        :param T:
        :param u:
        :param huge_constant: constant to impose the constraint ||w||_2 = 1
        :return:
        """
        # huge_constant is added as a new column to account for w norm constraint
        u = np.c_[u, huge_constant * np.ones(u.shape[0])]
        T = np.c_[T, huge_constant * np.ones(T.shape[0])]

        # Use non-negative least squares to solve the optimization problem
        w = np.array([scipy.optimize.nnls(A=T.T, b=u[i, :])[0] for i in range(u.shape[0])])
        return w
