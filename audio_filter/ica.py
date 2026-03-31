"""
ICA engine (FastICA) — built from first principles using NumPy.

Separates a multi-channel mixture X = A·S into independent sources S.

Pipeline:
    X  →  center  →  whiten (PCA sphering)  →  FastICA  →  S

Whitening:
    C  = (1/N) X Xᵀ
    C  = E D Eᵀ          (eigendecomposition, D diagonal)
    W_w = D^{-½} Eᵀ      (whitening matrix)
    Z  = W_w · X          (identity covariance, uncorrelated)

FastICA — symmetric fixed-point iteration (all components in parallel):
    Y      = W · Z
    W_new  = (1/N) g(Y) · Zᵀ  −  diag(mean(g′(Y))) · W
    W      = sym_orth(W_new)   via SVD: W = U·Vᵀ
    repeat until  max |diag(W_new · Wᵀ)| − 1  < tol

Nonlinearity (g = tanh, contrast function = log cosh):
    g (u) = tanh(u)
    g′(u) = 1 − tanh²(u)

tanh is well-suited to super-Gaussian (speech-like) sources.

Full unmixing matrix:
    W_full = W · W_w          maps original X → separated S
    S      = W_full · (X − mean)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def _center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract channel means.  X: (n_ch, n_samples)."""
    mean = X.mean(axis=1, keepdims=True)
    return X - mean, mean.squeeze(axis=1)


def _whiten(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Whiten X so the output has identity covariance.

    Parameters
    ----------
    X : (n_channels, n_samples) — zero-mean

    Returns
    -------
    Z      : (n_channels, n_samples) — whitened
    W_w    : (n_channels, n_channels) — whitening matrix
    """
    n_samples = X.shape[1]
    C = (X @ X.T) / n_samples                        # covariance

    eigenvalues, eigenvectors = np.linalg.eigh(C)    # ascending order

    # Sort descending and guard against tiny / negative eigenvalues
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = np.maximum(eigenvalues[idx], 1e-10)
    eigenvectors = eigenvectors[:, idx]

    W_w = np.diag(eigenvalues ** -0.5) @ eigenvectors.T
    Z   = W_w @ X
    return Z, W_w


# ---------------------------------------------------------------------------
# Symmetric orthogonalization
# ---------------------------------------------------------------------------

def _sym_orth(W: np.ndarray) -> np.ndarray:
    """
    Make rows of W orthonormal via SVD.

    W = U·S·Vᵀ  →  U·Vᵀ  (rows are orthonormal, magnitudes discarded).
    Equivalent to the standard formula  W (WᵀW)^{-½}.
    """
    U, _, Vt = np.linalg.svd(W, full_matrices=False)
    return U @ Vt


# ---------------------------------------------------------------------------
# FastICA core
# ---------------------------------------------------------------------------

def _fastica(
    Z: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
    random_state: int,
) -> tuple[np.ndarray, int]:
    """
    Symmetric FastICA on whitened data Z.

    Parameters
    ----------
    Z            : (n_channels, n_samples) — whitened, zero-mean
    n_components : number of sources to extract
    max_iter     : maximum fixed-point iterations
    tol          : convergence threshold on max |diag(W_new·Wᵀ)| − 1
    random_state : seed for reproducible initialisation

    Returns
    -------
    W     : (n_components, n_channels) — unmixing matrix in whitened space
    n_iter: number of iterations taken
    """
    n_channels, n_samples = Z.shape

    rng = np.random.default_rng(random_state)
    W   = _sym_orth(rng.standard_normal((n_components, n_channels)))

    for iteration in range(1, max_iter + 1):
        Y  = W @ Z                          # (n_components, n_samples)
        G  = np.tanh(Y)                     # g(Y)
        Gp = 1.0 - G ** 2                   # g′(Y) = 1 − tanh²

        # Fixed-point update
        W_new = (G @ Z.T) / n_samples  -  Gp.mean(axis=1, keepdims=True) * W

        # Symmetric orthogonalization
        W_new = _sym_orth(W_new)

        # Convergence: rows of W should be close to rows of W_new
        lim = float(np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0)))
        W   = W_new

        if lim < tol:
            break

    return W, iteration


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ica(
    X: np.ndarray,
    n_components: int | None = None,
    max_iter: int = 500,
    tol: float = 1e-5,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate a multi-channel mixture into independent components.

    Parameters
    ----------
    X            : (n_channels, n_samples) float array
                   Each row is one recorded channel (microphone).
    n_components : number of sources to recover (default = n_channels)
    max_iter     : FastICA iteration limit  (default 500)
    tol          : convergence tolerance    (default 1e-5)
    random_state : random seed for reproducibility

    Returns
    -------
    S      : (n_components, n_samples) — estimated source signals
    W_full : (n_components, n_channels) — full unmixing matrix
             satisfying  S ≈ W_full · (X − mean(X))
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[np.newaxis, :]

    if n_components is None:
        n_components = X.shape[0]

    X_c, _ = _center(X)
    Z, W_w = _whiten(X_c)

    W, _   = _fastica(Z, n_components, max_iter, tol, random_state)

    W_full = W @ W_w
    S      = W_full @ X_c

    return S, W_full
