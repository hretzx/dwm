"""
algorithms/kmeans.py
--------------------
K-Means clustering written from scratch (no scikit-learn).

Steps:
 1. Standardize every feature (z-score) so rainfall (hundreds) does not drown
    out pH (single digits).
 2. Pick k starting centroids (k-means++ spreads them out smartly).
 3. Repeat: assign each row to its nearest centroid, then move each centroid to
    the mean of its rows, until nothing changes.
 4. Run several random starts and keep the one with the lowest inertia
    (sum of squared distances to the centroids).
"""
import numpy as np


class KMeans:
    def __init__(self, k: int = 4, init: str = "kmeans++", max_iter: int = 100,
                 n_init: int = 5, seed: int = 42):
        self.k, self.init, self.max_iter = k, init, max_iter
        self.n_init, self.seed = n_init, seed

    def _standardize(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def _init_centroids(self, Z, rng):
        if self.init == "random":
            return Z[rng.choice(len(Z), self.k, replace=False)]
        # k-means++: first centroid random, the others chosen with probability
        # proportional to squared distance from the nearest centroid so far
        centroids = [Z[rng.integers(len(Z))]]
        for _ in range(self.k - 1):
            d2 = np.min([((Z - c) ** 2).sum(axis=1) for c in centroids], axis=0)
            centroids.append(Z[rng.choice(len(Z), p=d2 / d2.sum())])
        return np.array(centroids)

    def _run_once(self, Z, rng):
        centroids = self._init_centroids(Z, rng)
        for _ in range(self.max_iter):
            d = ((Z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = d.argmin(axis=1)
            new = np.array([Z[labels == j].mean(axis=0) if (labels == j).any() else centroids[j]
                            for j in range(self.k)])
            if np.allclose(new, centroids):
                break
            centroids = new
        d = ((Z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        return centroids, labels, float(d[np.arange(len(Z)), labels].sum())

    def fit(self, X) -> "KMeans":
        X = np.asarray(X, dtype=float)
        self.mean_, self.std_ = X.mean(axis=0), X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Z = self._standardize(X)
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            result = self._run_once(Z, rng)
            if best is None or result[2] < best[2]:
                best = result
        self.centroids_, self.labels_, self.inertia_ = best
        return self

    def predict(self, X) -> np.ndarray:
        """Assign new rows to the nearest centroid (in standardized space)."""
        Z = self._standardize(np.atleast_2d(X))
        d = ((Z[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1)

    def centroids_original(self) -> np.ndarray:
        """Centroids converted back to the original units (for profile tables)."""
        return self.centroids_ * self.std_ + self.mean_
