"""
Abstract class that defines the interface for all AA_* classes.

Note: notation used X ≈ A · B · X = A · Z

Author: Guillermo García Cobo
"""

import numpy as np
from abc import ABC, abstractmethod


class AA_Abstract(ABC):

    def __init__(self, n_archetypes: int, max_iter: int = 100, tol: float = 1e-6, verbose: bool = False):
        self.n_archetypes = n_archetypes
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.Z = None  # Archetypes
        self.n_samples, self.n_features = None, None
        self.RSS = None

    def fit(self, X: np.ndarray) -> 'AA_Abstract':
        """
        Computes the archetypes and the RSS from the data X, which are stored
        in the corresponding attributes
        :param X: data matrix, with shape (n_samples, n_features)
        :return: self
        """
        self.n_samples, self.n_features = X.shape
        self._fit(X)
        return self

    def _fit(self, X: np.ndarray):
        """
        Internal function that computes the archetypes and the RSS from the data X
        :param X: data matrix, with shape (n_samples, n_features)
        :return: None
        """
        # Initialize the archetypes
        B = np.eye(self.n_archetypes, self.n_samples)
        Z = B @ X

        A = np.eye(self.n_samples, self.n_archetypes)
        prev_RSS = None

        for _ in range(self.max_iter):
            A = self._computeA(X, Z, A)
            B = self._computeB(X, A, B)
            Z = B @ X
            RSS = self._rss(X, A, Z)
            if prev_RSS is not None and abs(prev_RSS - RSS) / prev_RSS < self.tol:
                break
            prev_RSS = RSS

        self.Z = Z
        self.RSS = RSS

    @staticmethod
    @abstractmethod
    def _computeA(X: np.ndarray, Z: np.ndarray, A: np.ndarray = None) -> np.ndarray:
        """
        Updates the A matrix given the data matrix X and the archetypes Z.
        A is the matrix that gives the best convex approximation of X by Z,
        so this function can be used during training and inference.
        For the latter, use the transform method instead
        :param X: data matrix, with shape (n_samples, n_features)
        :param Z: archetypes matrix, with shape (n_archetypes, n_features)
        :param A: A matrix, with shape (n_samples, n_archetypes)
        :return: A matrix, with shape (n_samples, n_archetypes)
        """
        pass

    @staticmethod
    @abstractmethod
    def _computeB(X: np.ndarray, A: np.ndarray, B: np.ndarray = None) -> np.ndarray:
        """
        Updates the B matrix given the data matrix X and the A matrix
        :param X: data matrix, with shape (n_samples, n_features)
        :param A: A matrix, with shape (n_samples, n_archetypes)
        :param B: B matrix, with shape (n_archetypes, n_samples)
        :return: B matrix, with shape (n_archetypes, n_samples)
        """
        pass

    def archetypes(self) -> np.ndarray:
        """
        Returns the archetypes' matrix
        :return: archetypes matrix, with shape (n_archetypes, n_features)
        """
        return self.Z

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the best convex approximation A of X by the archetypes
        :param X: data matrix, with shape (n_samples, n_features)
        :return: A matrix, with shape (n_samples, n_archetypes)
        """
        return self._computeA(X, self.Z)

    @staticmethod
    def _rss(X: np.ndarray, A: np.ndarray, Z: np.ndarray) -> float:
        """
        Computes the RSS of the data matrix X, given the A matrix and the archetypes Z
        :param X: data matrix, with shape (n_samples, n_features)
        :param A: A matrix, with shape (n_samples, n_archetypes)
        :param Z: archetypes matrix, with shape (n_archetypes, n_features)
        :return: RSS
        """
        return np.linalg.norm(X - A @ Z)**2
