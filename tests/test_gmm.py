"""
Tests for the GMM engine.

Run with:  python tests/test_gmm.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from audio_filter.gmm import (
    gmm_fit, gmm_log_prob, gmm_predict, gmm_bic, select_n_components
)


def make_mixture(n=600, seed=0):
    """
    3-component 2-D Gaussian mixture with well-separated clusters.
    Returns X (n, 2) and true labels (n,).
    """
    rng    = np.random.default_rng(seed)
    n_each = n // 3
    means  = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    Xs     = [rng.multivariate_normal(m, 0.3 * np.eye(2), n_each) for m in means]
    X      = np.vstack(Xs)
    labels = np.repeat(np.arange(3), n_each)
    return X, labels


# ── output structure ──────────────────────────────────────────────────────────
def test_fit_structure():
    X, _ = make_mixture()
    m = gmm_fit(X, n_components=3)
    assert "weights" in m and "means" in m and "covs" in m
    assert m["weights"].shape == (3,)
    assert m["means"].shape   == (3, 2)
    assert m["covs"].shape    == (3, 2)    # diag default
    assert abs(m["weights"].sum() - 1.0) < 1e-9
    print(f"[PASS] fit structure  weights_sum={m['weights'].sum():.6f}")


# ── weights sum to 1 ──────────────────────────────────────────────────────────
def test_weights_sum():
    for k in [1, 2, 5]:
        m = gmm_fit(make_mixture()[0], n_components=k)
        assert abs(m["weights"].sum() - 1.0) < 1e-9, f"k={k} weights sum={m['weights'].sum()}"
    print("[PASS] weights sum to 1 for k=1,2,5")


# ── diag covariances are positive ─────────────────────────────────────────────
def test_diag_positive():
    X, _ = make_mixture()
    m = gmm_fit(X, n_components=3, covariance_type="diag")
    assert np.all(m["covs"] > 0), f"Negative diag variance: {m['covs'].min()}"
    print(f"[PASS] diag covs positive  min={m['covs'].min():.4e}")


# ── full covariances are symmetric positive-definite ─────────────────────────
def test_full_spd():
    X, _ = make_mixture()
    m = gmm_fit(X, n_components=3, covariance_type="full")
    assert m["covs"].shape == (3, 2, 2)
    for j in range(3):
        C   = m["covs"][j]
        err = np.max(np.abs(C - C.T))
        assert err < 1e-12,    f"Cov {j} not symmetric: {err:.2e}"
        evals = np.linalg.eigvalsh(C)
        assert np.all(evals > 0), f"Cov {j} not PD: evals={evals}"
    print("[PASS] full covs symmetric PD")


# ── log_prob returns finite values ────────────────────────────────────────────
def test_log_prob_finite():
    X, _ = make_mixture()
    m    = gmm_fit(X, n_components=3)
    lp   = gmm_log_prob(X, m)
    assert lp.shape == (len(X),)
    assert np.all(np.isfinite(lp)), f"Non-finite log probs: {np.sum(~np.isfinite(lp))}"
    print(f"[PASS] log_prob finite  mean={lp.mean():.2f}")


# ── cluster recovery on well-separated mixture ────────────────────────────────
def test_cluster_recovery():
    """
    With 3 tight well-separated clusters, accuracy (after permutation matching)
    should be > 95 %.
    """
    X, true_labels = make_mixture()
    m      = gmm_fit(X, n_components=3, random_state=0)
    pred   = gmm_predict(X, m)

    # Greedy permutation matching
    best_acc = 0.0
    from itertools import permutations
    for perm in permutations(range(3)):
        mapped = np.array([perm[p] for p in pred])
        acc    = np.mean(mapped == true_labels)
        best_acc = max(best_acc, acc)

    assert best_acc > 0.95, f"Cluster recovery accuracy too low: {best_acc:.3f}"
    print(f"[PASS] cluster recovery  acc={best_acc:.3f}")


# ── log-likelihood increases monotonically during EM ─────────────────────────
def test_log_lik_monotone():
    """
    Fit with max_iter=1, 2, … and check that log-likelihood never decreases.
    """
    X, _ = make_mixture()
    liks = []
    for iters in range(1, 21):
        m = gmm_fit(X, n_components=3, max_iter=iters, tol=0.0, random_state=0)
        liks.append(m["log_lik"])

    for i in range(1, len(liks)):
        assert liks[i] >= liks[i-1] - 1e-9, (
            f"LL decreased at iter {i+1}: {liks[i-1]:.6f} -> {liks[i]:.6f}"
        )
    print(f"[PASS] LL monotone  L1={liks[0]:.4f}  L20={liks[-1]:.4f}")


# ── BIC: lower for true k than for under/over-fit ────────────────────────────
def test_bic_selects_true_k():
    X, _ = make_mixture()
    bics = {}
    for k in [1, 2, 3, 4, 5]:
        m       = gmm_fit(X, n_components=k, random_state=0)
        bics[k] = gmm_bic(X, m)

    best_k = min(bics, key=bics.get)
    assert best_k == 3, (
        f"BIC selected k={best_k}, expected 3.  BICs={bics}"
    )
    print(f"[PASS] BIC selects k=3  BICs={{{', '.join(f'{k}:{v:.0f}' for k,v in sorted(bics.items()))}}}")


# ── select_n_components returns correct shape ─────────────────────────────────
def test_select_n_components():
    X, _ = make_mixture()
    best_k, best_model, bics = select_n_components(X, range(1, 6), random_state=0)
    assert len(bics) == 5
    assert best_k == best_model["means"].shape[0]
    assert all(len(pair) == 2 for pair in bics)
    print(f"[PASS] select_n_components  best_k={best_k}  n_bics={len(bics)}")


# ── 1-D input doesn't crash ───────────────────────────────────────────────────
def test_1d_input():
    rng = np.random.default_rng(42)
    X   = np.concatenate([rng.normal(0, 1, 200), rng.normal(5, 1, 200)])
    m   = gmm_fit(X, n_components=2)
    assert m["weights"].shape == (2,)
    assert m["means"].shape   == (2, 1)
    lp  = gmm_log_prob(X, m)
    assert np.all(np.isfinite(lp))
    print("[PASS] 1-D input OK")


if __name__ == "__main__":
    test_fit_structure()
    test_weights_sum()
    test_diag_positive()
    test_full_spd()
    test_log_prob_finite()
    test_cluster_recovery()
    test_log_lik_monotone()
    test_bic_selects_true_k()
    test_select_n_components()
    test_1d_input()
    print("\nAll tests passed.")
