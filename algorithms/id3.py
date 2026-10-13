"""
algorithms/id3.py
-----------------
ID3 decision tree written from scratch (no scikit-learn).

How it works, in plain language:
 1. ID3 only understands categories, so every continuous feature (N, P, K,
    temperature ...) is first cut into equal-frequency bins (quantile bins).
 2. At each node we measure how mixed the crop labels are using ENTROPY.
 3. We pick the feature whose split reduces entropy the most. That reduction
    is the INFORMATION GAIN.
 4. We repeat on every branch until the node is pure, we run out of features,
    or we hit the depth limit.
"""
import numpy as np
import pandas as pd

# Readable names for the bins of each feature, depending on how many bins exist
LEVEL_NAMES = {
    1: ["all values"],
    2: ["low", "high"],
    3: ["low", "medium", "high"],
    4: ["very low", "low", "high", "very high"],
    5: ["very low", "low", "medium", "high", "very high"],
}


def entropy(y: np.ndarray) -> float:
    """Entropy = -sum(p * log2 p) over the class proportions p."""
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y)
    p = counts[counts > 0] / len(y)
    return float(-(p * np.log2(p)).sum())


class Node:
    """One node of the tree. A node with no children is a leaf."""

    def __init__(self, counts: np.ndarray):
        self.counts = counts            # crop counts of training rows at this node
        self.n = int(counts.sum())      # number of training rows at this node
        self.label = int(counts.argmax())  # majority class (the prediction at a leaf)
        self.feature = None             # feature index used to split (None = leaf)
        self.gain = 0.0                 # information gain of that split
        self.children = {}              # bin index -> child Node


class ID3Tree:
    def __init__(self, n_bins: int = 5, max_depth: int = 6, min_samples: int = 5):
        self.n_bins = n_bins
        self.max_depth = max_depth
        self.min_samples = min_samples

    # ------------------------------------------------------------ discretizing
    def _bin_col(self, feature: str, values) -> np.ndarray:
        """Turn continuous values into bin numbers 0..m using the stored edges."""
        return np.searchsorted(self.edges[feature], np.asarray(values, dtype=float), side="right")

    def bin_label(self, feature: str, b: int) -> tuple[str, str]:
        """Return (level name, numeric range text) for bin number b."""
        edges = self.edges[feature]
        m = len(edges)
        names = LEVEL_NAMES.get(m + 1, [f"bin {i + 1}" for i in range(m + 1)])
        if b == 0:
            rng = f"≤ {edges[0]:.4g}"
        elif b == m:
            rng = f"> {edges[-1]:.4g}"
        else:
            rng = f"{edges[b - 1]:.4g} to {edges[b]:.4g}"
        return names[b], rng

    # ---------------------------------------------------------------- training
    def fit(self, df: pd.DataFrame, features: list[str], target: str) -> "ID3Tree":
        self.features = list(features)
        self.classes_ = sorted(df[target].unique())
        class_index = {c: i for i, c in enumerate(self.classes_)}
        y = df[target].map(class_index).to_numpy()

        # Equal-frequency bin edges learned from the training data
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = {f: np.unique(np.quantile(df[f].to_numpy(float), qs)) for f in self.features}

        X = np.column_stack([self._bin_col(f, df[f]) for f in self.features])
        self.root = self._build(X, y, set(range(len(self.features))), depth=0)
        return self

    def _build(self, X, y, available: set, depth: int) -> Node:
        node = Node(np.bincount(y, minlength=len(self.classes_)))

        # Stop: pure node, no features left, too deep, or too few rows
        if (len(np.unique(y)) == 1 or not available
                or depth >= self.max_depth or len(y) < self.min_samples):
            return node

        # Pick the feature with the highest information gain
        base = entropy(y)
        best_f, best_gain = None, 1e-9
        for f in available:
            remainder = 0.0
            for b in np.unique(X[:, f]):
                mask = X[:, f] == b
                remainder += mask.sum() / len(y) * entropy(y[mask])
            gain = base - remainder
            if gain > best_gain:
                best_f, best_gain = f, gain
        if best_f is None:          # no split helps
            return node

        node.feature, node.gain = best_f, best_gain
        for b in np.unique(X[:, best_f]):
            mask = X[:, best_f] == b
            node.children[int(b)] = self._build(X[mask], y[mask], available - {best_f}, depth + 1)
        return node

    # -------------------------------------------------------------- prediction
    def predict_with_path(self, row: dict) -> tuple[str, list[dict], float]:
        """
        Classify one input and record every decision taken.
        Returns (crop, path, leaf_purity). Each path step says which feature was
        tested, the input's value, the bin it fell into, and how many training
        rows reached that node.
        """
        node, path = self.root, []
        while node.feature is not None:
            f = self.features[node.feature]
            b = int(self._bin_col(f, [row[f]])[0])
            level, rng = self.bin_label(f, b)
            path.append({
                "feature": f, "value": float(row[f]), "level": level, "range": rng,
                "gain": node.gain, "rows_here": node.n,
            })
            if b not in node.children:   # bin never seen at this node in training
                break
            node = node.children[b]
        purity = float(node.counts[node.label] / node.n)
        return self.classes_[node.label], path, purity

    def predict(self, df: pd.DataFrame) -> list[str]:
        """Predict a whole DataFrame (used for the accuracy check)."""
        return [self.predict_with_path(r)[0] for r in df[self.features].to_dict("records")]
