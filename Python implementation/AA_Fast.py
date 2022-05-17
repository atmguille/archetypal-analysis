"""
Implementation of Frank-Wolfe algorithm for the archetypal analysis.

Algorithm is based on the paper: "Archetypal Analysis as an Autoencoder"
(https://www.researchgate.net/publication/282733207_Archetypal_Analysis_as_an_Autoencoder)
Code adapted from https://github.com/nichohelmut/football_results/blob/master/clustering/clustering.py

Author: Guillermo García Cobo
"""

import numpy as np

from AA_Abstract import AA_Abstract


class AA_Fast(AA_Abstract):

    def __init__(self, n_archetypes: int, max_iter: int = 100, tol: float = 1e-6, verbose: bool = False,
                 derivative_max_iter: int = 10):
        super().__init__(n_archetypes, max_iter, tol, verbose)
        self.derivative_max_iter = derivative_max_iter

    def _computeA(self, X: np.ndarray, Z: np.ndarray, A: np.ndarray = None) -> np.ndarray:
        A = np.zeros((self.n_samples, self.n_archetypes))
        A[:, 0] = 1.0
        e = np.zeros(A.shape)
        for t in range(self.derivative_max_iter):
            # brackets are VERY important to save time
            G = 2.0 * (A @ (Z @ Z.T) - X @ Z.T)
            # Get the argument mins along each column
            argmins = np.argmin(G, axis=1)
            e[range(self.n_samples), argmins] = 1.0
            A += 2.0 / (t + 2.0) * (e - A)
            e[range(self.n_samples), argmins] = 0.0
        return A

    def _computeB(self, X: np.ndarray, A: np.ndarray, B: np.ndarray = None) -> np.ndarray:
        B = np.zeros((self.n_archetypes, self.n_samples))
        B[:, 0] = 1.0
        e = np.zeros(B.shape)
        for t in range(self.derivative_max_iter):
            # brackets are VERY important to save time
            t1 = (A.T @ A) @ (B @ X) @ X.T
            t2 = (A.T @ X) @ X.T
            G = 2.0 * (t1 - t2)
            argmins = np.argmin(G, axis=1)
            e[range(self.n_archetypes), argmins] = 1.0
            B += 2.0 / (t + 2.0) * (e - B)
            e[range(self.n_archetypes), argmins] = 0.0
        return B
