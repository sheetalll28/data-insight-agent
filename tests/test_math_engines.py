"""Mathematical correctness of the from-scratch engines vs trusted references."""
import numpy as np
import pandas as pd
import pytest

from outlier_detection import calculate_iqr_bounds
from pipeline import calculate_correlation_matrix
from nn_from_scratch import (NeuralNetwork, permutation_importance, r2_score,
                             train_test_split_indices, minmax_fit, minmax_transform)
from kmeans_from_scratch import KMeansFromScratch
from levenshtein import levenshtein_distance, string_similarity


# ---------------- IQR ----------------
@pytest.mark.parametrize("n", [1, 2, 5, 7, 10, 101, 1000])
def test_iqr_matches_numpy_linear_percentiles(n):
    x = np.random.default_rng(n).normal(size=n)
    q1, q3 = np.percentile(x, [25, 75])
    lo, hi = calculate_iqr_bounds(x)
    assert lo == pytest.approx(q1 - 1.5 * (q3 - q1))
    assert hi == pytest.approx(q3 + 1.5 * (q3 - q1))


def test_iqr_ignores_nan_and_handles_empty():
    assert calculate_iqr_bounds(np.array([])) == (0, 0)
    lo, hi = calculate_iqr_bounds(np.array([1, 2, 3, 4, np.nan]))
    q1, q3 = np.percentile([1, 2, 3, 4], [25, 75])
    assert (lo, hi) == pytest.approx((q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)))


# ---------------- Pearson ----------------
def test_pearson_matches_numpy_corrcoef():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    X[:, 1] = 2 * X[:, 0] + rng.normal(scale=0.1, size=300)
    df = pd.DataFrame(X, columns=list("abcdef"))
    np.testing.assert_allclose(calculate_correlation_matrix(df).values, np.corrcoef(X, rowvar=False), atol=1e-12)


def test_pearson_constant_column_is_zero_not_nan():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 5, 5, 5]})
    m = calculate_correlation_matrix(df)
    assert m.loc["a", "b"] == 0.0 and not m.isna().any().any()


def test_pearson_encodes_categoricals_like_original():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "c": ["a", "b", "a", "b"]})
    m = calculate_correlation_matrix(df)
    assert m.loc["x", "c"] == pytest.approx(np.corrcoef([1, 2, 3, 4], [0, 1, 0, 1])[0, 1])


# ---------------- Neural network ----------------
def test_backprop_matches_finite_differences():
    X = np.random.default_rng(1).standard_normal((20, 4))
    Y = np.random.default_rng(2).standard_normal((20, 1))
    nn = NeuralNetwork(4, 6, 1, seed=3)
    d1, d2 = nn.gradient_check(X, Y)
    assert d1 < 1e-6 and d2 < 1e-6


def test_network_does_not_touch_global_rng():
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    NeuralNetwork(3, 4, 1)
    assert np.random.rand() == expected


def test_network_learns_and_importance_finds_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5))
    y = 3 * X[:, 0] + np.sin(2 * X[:, 1]) + 0.3 * rng.normal(size=600)
    tr, te = train_test_split_indices(len(X))
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == len(X)
    mn, mx = minmax_fit(X[tr])
    Xtr, Xte = minmax_transform(X[tr], mn, mx), minmax_transform(X[te], mn, mx)
    ymin, ymax = y[tr].min(), y[tr].max()
    ytr = ((y[tr] - ymin) / (ymax - ymin)).reshape(-1, 1)
    yte = ((y[te] - ymin) / (ymax - ymin)).reshape(-1, 1)
    nn = NeuralNetwork(5, 8, 1, learning_rate=0.1).fit(Xtr, ytr, epochs=1000)
    assert r2_score(yte, nn.predict(Xte)) > 0.9
    imp, _ = permutation_importance(nn.predict, Xte, yte)
    assert np.argmax(imp) == 0                 # the dominant feature
    assert imp[0] > 10 * max(imp[2:].max(), 1e-12)  # noise features ~0


# ---------------- K-Means ----------------
def test_kmeans_matches_sklearn_inertia_on_blobs():
    sk = pytest.importorskip("sklearn")
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score
    X, y = make_blobs(2000, 4, centers=5, cluster_std=1.0, random_state=1)
    m = KMeansFromScratch(k=5, n_init=5)
    labels = m.fit(X)
    ref = KMeans(5, n_init=10, random_state=0).fit(X)
    assert m.wcss == pytest.approx(ref.inertia_, rel=1e-6)
    assert adjusted_rand_score(y, labels) > 0.99
    np.testing.assert_array_equal(m.predict(X), labels)


def test_kmeans_wcss_is_consistent_and_reproducible():
    X = np.random.default_rng(0).normal(size=(200, 3))
    a, b = KMeansFromScratch(k=3, seed=7), KMeansFromScratch(k=3, seed=7)
    la, lb = a.fit(X), b.fit(X)
    np.testing.assert_array_equal(la, lb)
    assert a.wcss == pytest.approx(np.sum((X - a.centroids[la]) ** 2))
    assert len(np.unique(la)) == 3


def test_kmeans_rejects_k_larger_than_rows_and_handles_duplicates():
    with pytest.raises(ValueError):
        KMeansFromScratch(k=5).fit(np.zeros((3, 2)))
    labels = KMeansFromScratch(k=2, n_init=2).fit(np.zeros((10, 2)))  # all identical points
    assert labels.shape == (10,)


# ---------------- Levenshtein ----------------
def _reference_lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@pytest.mark.parametrize("a,b", [("CAT", "COAT"), ("kitten", "sitting"), ("Depression", "MDD"),
                                 ("Major Depression", "Major Depressive Disorder"),
                                 ("schizophrenia", "Schizo"), ("", "abc"), ("same", "same")])
def test_levenshtein_matches_reference(a, b):
    assert levenshtein_distance(a, b) == _reference_lev(a, b)
    assert 0.0 <= string_similarity(a, b) <= 1.0
