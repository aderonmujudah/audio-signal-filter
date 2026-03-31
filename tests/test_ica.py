"""
Tests for the ICA engine (FastICA).

Run with:  python tests/test_ica.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from audio_filter.ica import _center, _whiten, _sym_orth, ica

SR     = 16000
N      = SR * 2   # 2 seconds


def make_sources(n=2, length=N, seed=0):
    """Two statistically independent non-Gaussian sources."""
    rng = np.random.default_rng(seed)
    # Super-Gaussian (speech-like): sin of increasing frequency + sparse noise
    t   = np.arange(length) / SR
    s1  = np.sin(2 * np.pi * 440 * t) + 0.3 * rng.standard_normal(length)
    s2  = np.sign(np.sin(2 * np.pi * 180 * t))   # square wave — sparse
    S   = np.vstack([s1, s2]).astype(np.float64)
    S  /= S.std(axis=1, keepdims=True)
    return S


def mix(S, seed=1):
    """Random invertible mixing matrix."""
    rng = np.random.default_rng(seed)
    A   = rng.standard_normal((S.shape[0], S.shape[0]))
    while abs(np.linalg.det(A)) < 0.1:      # ensure invertible
        A = rng.standard_normal((S.shape[0], S.shape[0]))
    return A @ S, A


# ── centering removes mean ────────────────────────────────────────────────────
def test_center():
    X    = np.random.default_rng(0).standard_normal((2, 1000)) + 5.0
    Xc, mean = _center(X)
    assert np.allclose(Xc.mean(axis=1), 0, atol=1e-12)
    assert np.allclose(mean, X.mean(axis=1), atol=1e-12)
    print("[PASS] centering removes mean")


# ── whitened data has identity covariance ────────────────────────────────────
def test_whitening():
    rng = np.random.default_rng(0)
    X   = rng.standard_normal((3, 5000))
    X  -= X.mean(axis=1, keepdims=True)
    Z, W_w = _whiten(X)

    cov = (Z @ Z.T) / Z.shape[1]
    err = np.max(np.abs(cov - np.eye(3)))
    assert err < 1e-10, f"Whitened covariance not identity: max err={err:.2e}"
    print(f"[PASS] whitening  cov_err={err:.2e}")


# ── sym_orth produces orthonormal rows ───────────────────────────────────────
def test_sym_orth():
    rng = np.random.default_rng(0)
    W   = rng.standard_normal((3, 3))
    Wo  = _sym_orth(W)
    gram = Wo @ Wo.T
    err  = np.max(np.abs(gram - np.eye(3)))
    assert err < 1e-12, f"sym_orth rows not orthonormal: {err:.2e}"
    print(f"[PASS] sym_orth  gram_err={err:.2e}")


# ── output shape ─────────────────────────────────────────────────────────────
def test_ica_shape():
    S     = make_sources()
    X, _  = mix(S)
    Shat, W_full = ica(X)
    assert Shat.shape   == S.shape,          f"S shape: {Shat.shape} vs {S.shape}"
    assert W_full.shape == (S.shape[0],) * 2, f"W shape: {W_full.shape}"
    print(f"[PASS] ica shape  S={Shat.shape}  W={W_full.shape}")


# ── W_full is invertible (det != 0) ──────────────────────────────────────────
def test_unmixing_invertible():
    S, _  = make_sources(), None
    X, _  = mix(make_sources())
    _, W  = ica(X)
    det   = abs(np.linalg.det(W))
    assert det > 1e-3, f"|det(W_full)| too small: {det:.4e}"
    print(f"[PASS] unmixing invertible  |det|={det:.4f}")


# ── recovered sources are more independent than the mixture ──────────────────
def test_source_separation_quality():
    """
    After ICA, |corr(S_hat[0], S_hat[1])| should be much smaller than
    |corr(X[0], X[1])| — cross-correlation falls when sources are separated.
    """
    S        = make_sources()
    X, A     = mix(S)

    corr_X   = abs(np.corrcoef(X)[0, 1])
    Shat, _  = ica(X)
    corr_S   = abs(np.corrcoef(Shat)[0, 1])

    assert corr_S < corr_X, (
        f"ICA did not reduce correlation: corr(X)={corr_X:.4f}  corr(Shat)={corr_S:.4f}"
    )
    print(f"[PASS] separation quality  corr(X)={corr_X:.4f}  ->  corr(Shat)={corr_S:.4f}")


# ── W_full @ A is close to a signed permutation matrix ───────────────────────
def test_permutation_recovery():
    """
    W · A should be a scaled permutation matrix:
    exactly one large entry per row and column.
    """
    S        = make_sources()
    X, A     = mix(S)
    _, W     = ica(X)

    P        = W @ A
    # Normalize rows so max element = 1
    P_norm   = P / np.abs(P).max(axis=1, keepdims=True)

    # Each row should have exactly one entry with |value| ≈ 1
    row_maxes = np.sort(np.abs(P_norm), axis=1)[:, ::-1]
    # Largest value should dominate the second-largest by at least 5x
    ratio = row_maxes[:, 0] / np.maximum(row_maxes[:, 1], 1e-8)
    assert np.all(ratio > 5), f"W·A not permutation-like, ratios={ratio}"
    print(f"[PASS] W·A permutation  row ratios={np.round(ratio, 1)}")


# ── single-channel input doesn't crash ───────────────────────────────────────
def test_single_channel():
    rng = np.random.default_rng(0)
    x   = rng.standard_normal(SR)
    S, W = ica(x[np.newaxis, :], n_components=1)
    assert S.shape == (1, SR)
    print("[PASS] single channel ICA shape OK")


if __name__ == "__main__":
    test_center()
    test_whitening()
    test_sym_orth()
    test_ica_shape()
    test_unmixing_invertible()
    test_source_separation_quality()
    test_permutation_recovery()
    test_single_channel()
    print("\nAll tests passed.")
