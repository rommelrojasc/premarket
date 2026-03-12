"""Combinatorial Purged Cross-Validation (CPCV) for financial time series."""
import numpy as np
from itertools import combinations
from sklearn.model_selection import BaseCrossValidator


class CombinatorialPurgedCV(BaseCrossValidator):
    """
    CPCV: Combinatorial Purged Cross-Validation.

    Unlike standard k-fold, CPCV:
    1. Creates all C(n_splits, n_test_groups) train/test combinations
    2. Purges training samples near the test boundary to prevent leakage
    3. Adds an embargo period after test groups

    Reference: De Prado, "Advances in Financial Machine Learning" (2018)
    """

    def __init__(self, n_splits=10, n_test_groups=2, purge_gap=5, embargo_gap=2):
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap

    def get_n_splits(self, X=None, y=None, groups=None):
        from math import comb
        return comb(self.n_splits, self.n_test_groups)

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Create roughly equal groups
        group_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        remainder = n_samples % self.n_splits
        group_sizes[:remainder] += 1

        group_starts = np.cumsum(np.concatenate([[0], group_sizes[:-1]]))
        group_ends = np.cumsum(group_sizes)

        # Generate all combinations of test groups
        for test_group_indices in combinations(range(self.n_splits), self.n_test_groups):
            test_indices = []
            purge_indices = set()

            for gi in test_group_indices:
                start = group_starts[gi]
                end = group_ends[gi]
                test_indices.extend(indices[start:end])

                # Purge: remove training samples close to test boundaries
                purge_start = max(0, start - self.purge_gap)
                purge_end_val = min(n_samples, end + self.embargo_gap)

                for pi in range(purge_start, start):
                    purge_indices.add(pi)
                for pi in range(end, purge_end_val):
                    purge_indices.add(pi)

            test_set = set(test_indices)
            train_indices = [i for i in indices if i not in test_set and i not in purge_indices]

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield np.array(train_indices), np.array(test_indices)


def evaluate_with_cpcv(model_class, X, y, n_splits=10, n_test_groups=2,
                       purge_gap=5, embargo_gap=2, **model_kwargs):
    """
    Evaluate a model using CPCV and return aggregated metrics.

    Returns dict with accuracy, per-fold results, and backtest paths.
    """
    from sklearn.metrics import accuracy_score, log_loss
    cv = CombinatorialPurgedCV(n_splits, n_test_groups, purge_gap, embargo_gap)

    fold_results = []
    all_preds = np.zeros(len(y))
    all_probs = np.zeros(len(y))
    pred_counts = np.zeros(len(y))

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        model = model_class(**model_kwargs)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        probs = model.predict_proba(X[test_idx])

        bull_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

        acc = accuracy_score(y[test_idx], preds)
        fold_results.append({
            "fold": fold_idx,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": acc,
        })

        all_preds[test_idx] += preds
        all_probs[test_idx] += bull_prob
        pred_counts[test_idx] += 1

    # Average predictions across folds where each sample appeared in test
    mask = pred_counts > 0
    avg_probs = np.zeros(len(y))
    avg_probs[mask] = all_probs[mask] / pred_counts[mask]
    avg_preds = (avg_probs >= 0.5).astype(int)

    overall_acc = accuracy_score(y[mask], avg_preds[mask])
    mean_fold_acc = np.mean([f["accuracy"] for f in fold_results])
    std_fold_acc = np.std([f["accuracy"] for f in fold_results])

    return {
        "overall_accuracy": overall_acc,
        "mean_fold_accuracy": mean_fold_acc,
        "std_fold_accuracy": std_fold_acc,
        "n_folds": len(fold_results),
        "fold_results": fold_results,
    }
