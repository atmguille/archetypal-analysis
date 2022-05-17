"""
Implementation of the AA Principal Convex Hull algorithm.

Algorithm is based on the paper: "Archetypal analysis for machine learning and data mining"
(https://www.sciencedirect.com/science/article/pii/S0925231211006060)

Author: Guillermo García Cobo
"""


import numpy as np

from AA_Abstract import AA_Abstract


class AA_PCHA(AA_Abstract):

    def __init__(self, n_archetypes: int, max_iter: int = 100, tol: float = 1e-6, verbose: bool = False,
                 derivative_max_iter: int = 10):
        super().__init__(n_archetypes, max_iter, tol, verbose)
        self.derivative_max_iter = derivative_max_iter
        self.muA = 1
        self.muB = 1

    def _computeA(self, X: np.ndarray, Z: np.ndarray, A: np.ndarray) -> np.ndarray:
        prev_RSS = self._rss(X, A, Z)

        e = np.ones(self.n_archetypes)

        for t in range(self.derivative_max_iter):
            # brackets are VERY important to save time
            # [G] ~  n x k
            G = 2.0 * (A @ (Z @ Z.T) - X @ Z.T)
            G = G - e * np.sum(A * G, axis=1, keepdims=True)  # chain rule of projection

            prev_A = A
            for t in range(self.derivative_max_iter*100):  # Base implementation has a while True
                A = (prev_A - self.muA * G).clip(min=0)
                A = A / (np.sum(A, axis=1, keepdims=True) + np.finfo(float).eps)  # Avoid division by zero
                RSS = self._rss(X, A, Z)
                if RSS <= prev_RSS * (1+self.tol):
                    self.muA *= 1.2
                    break
                else:
                    self.muA /= 2.0

        return A

    def _computeB(self, X: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        prev_RSS = self._rss(X, A, B @ X)

        e = np.ones(self.n_samples)

        for t in range(self.derivative_max_iter):
            # brackets are VERY important to save time
            t1 = (A.T @ A) @ (B @ X) @ X.T
            t2 = (A.T @ X) @ X.T
            G = 2.0 * (t1 - t2)
            G = G - e * np.sum(B * G, axis=1, keepdims=True)  # chain rule of projection
            prev_B = B
            for t in range(self.derivative_max_iter*100):  # Base implementation has a while True
                B = (prev_B - self.muB * G).clip(min=0)
                B = B / (np.sum(B, axis=1, keepdims=True) + np.finfo(float).eps)  # Avoid division by zero
                Z = B @ X
                RSS = self._rss(X, A, Z)
                if RSS <= prev_RSS * (1+self.tol):
                    self.muB *= 1.2
                    break
                else:
                    self.muB /= 2.0
        return B
