"""
Gaussian Mixture Model (GMM) with EM training + BIC model selection.

Model:   p(x) = Σ_k  π_k · N(x ; μ_k , Σ_k)

EM algorithm
────────────
E-step (log-space for numerical stability):

    log r_nk  =  log π_k  +  log N(x_n ; μ_k , Σ_k)
    log r_nk  -=  logsumexp_k(log r_nk)          ← normalize to responsibilities

M-step:

    N_k   =  Σ_n  r_nk
    π_k   =  N_k / N
    μ_k   =  (1/N_k)  Σ_n  r_nk · x_n
    Σ_k   =  (1/N_k)  Σ_n  r_nk · (x_n−μ_k)(x_n−μ_k)ᵀ  +  reg·I

Covariance types
────────────────
'diag'  — Σ_k = diag(σ²_{k1}, …, σ²_{kd}).  Stores (k, d).
          Standard for speech (MFCCs are approximately decorrelated).
'full'  — Full symmetric positive-definite matrix.  Stores (k, d, d).
          Evaluated via Cholesky for numerical stability.

BIC model selection
────────────────────
    BIC(k) = −2 · ℓ(k)  +  p(k) · log N

    ℓ(k) = total log-likelihood under the fitted k-component model
    N     = number of samples
    p(k)  = number of free parameters:
              diag  →  k·(2d + 1) − 1
              full  →  k·(d + d(d+1)/2) + (k − 1)

    Lower BIC is preferred.  select_n_components() returns the k in a
    candidate range with the lowest BIC.
"""

import numpy as np


_EPS    = 1e-10
_LOG2PI = np.log(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Numerics helpers
# ---------------------------------------------------------------------------

def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log-sum-exp along `axis`."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out   = np.log(np.sum(np.exp(a - a_max), axis=axis)) + a_max.squeeze(axis=axis)
    return out


def _log_gauss_diag(
    X: np.ndarray,      # (n, d)
    mean: np.ndarray,   # (d,)
    var: np.ndarray,    # (d,)  diagonal variances
) -> np.ndarray:        # (n,)
    """Log N(x ; mean, diag(var)) for each row of X."""
    d    = X.shape[1]
    diff = X - mean                              # (n, d)
    return -0.5 * (
        d * _LOG2PI
        + np.sum(np.log(np.maximum(var, _EPS)))
        + np.sum(diff ** 2 / np.maximum(var, _EPS), axis=1)
    )


def _log_gauss_full(
    X: np.ndarray,      # (n, d)
    mean: np.ndarray,   # (d,)
    cov: np.ndarray,    # (d, d)
) -> np.ndarray:        # (n,)
    """Log N(x ; mean, cov) for each row of X, via Cholesky."""
    d    = X.shape[1]
    diff = X - mean                              # (n, d)
    L    = np.linalg.cholesky(cov)              # (d, d) lower triangular
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    y    = np.linalg.solve(L, diff.T)           # (d, n)
    mah  = np.sum(y ** 2, axis=0)               # (n,)
    return -0.5 * (d * _LOG2PI + log_det + mah)


# ---------------------------------------------------------------------------
# E-step and M-step
# ---------------------------------------------------------------------------

def _e_step(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    cov_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute log responsibilities and per-sample log-likelihoods.

    Returns
    -------
    log_resp : (n, k)  normalized log responsibilities
    log_liks : (n,)    log p(x_n) under the mixture
    """
    n, _ = X.shape
    k    = len(weights)
    lp   = np.empty((n, k))

    for j in range(k):
        lw = np.log(max(weights[j], _EPS))
        if cov_type == "diag":
            lp[:, j] = lw + _log_gauss_diag(X, means[j], covs[j])
        else:
            lp[:, j] = lw + _log_gauss_full(X, means[j], covs[j])

    log_liks = _logsumexp(lp, axis=1)              # (n,)
    log_resp = lp - log_liks[:, np.newaxis]        # (n, k)
    return log_resp, log_liks


def _m_step(
    X: np.ndarray,
    log_resp: np.ndarray,
    cov_type: str,
    reg: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Re-estimate weights, means, covariances from soft responsibilities.
    `reg` is added to variances / covariance diagonals for numerical stability.
    """
    n, d = X.shape
    k    = log_resp.shape[1]
    resp = np.exp(log_resp)                        # (n, k)
    Nk   = np.maximum(resp.sum(axis=0), _EPS)     # (k,)

    weights = Nk / n
    means   = (resp.T @ X) / Nk[:, np.newaxis]    # (k, d)

    if cov_type == "diag":
        covs = np.empty((k, d))
        for j in range(k):
            diff     = X - means[j]
            covs[j]  = (resp[:, j] @ (diff ** 2)) / Nk[j] + reg
    else:
        covs = np.empty((k, d, d))
        for j in range(k):
            diff     = X - means[j]                # (n, d)
            covs[j]  = (resp[:, j:j+1] * diff).T @ diff / Nk[j]
            covs[j] += reg * np.eye(d)

    return weights, means, covs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gmm_fit(
    X: np.ndarray,
    n_components: int,
    covariance_type: str = "diag",
    max_iter: int = 100,
    tol: float = 1e-4,
    reg: float = 1e-6,
    random_state: int = 0,
    n_init: int = 3,
) -> dict:
    """
    Fit a GMM with `n_components` Gaussians via EM.

    Runs `n_init` restarts with different random seeds and returns the
    result with the highest log-likelihood (avoids bad local minima).

    Parameters
    ----------
    X              : (n_samples, n_features)
    n_components   : number of mixture components k
    covariance_type: 'diag' (default) or 'full'
    max_iter       : EM iteration limit
    tol            : stop when |Δ avg log-likelihood| < tol
    reg            : covariance regularisation (added to diagonal)
    random_state   : base seed; restart i uses seed random_state + i
    n_init         : number of random restarts (default 3)

    Returns
    -------
    dict with keys:
        weights  (k,)
        means    (k, d)
        covs     (k, d) if diag  |  (k, d, d) if full
        cov_type str
        log_lik  float  — final average per-sample log-likelihood
        n_iter   int
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, np.newaxis]

    n, d = X.shape
    k    = n_components

    if k < 1:
        raise ValueError(f"n_components must be ≥ 1, got {k}")
    if k > n:
        raise ValueError(f"n_components ({k}) must be ≤ n_samples ({n})")
    if covariance_type not in ("diag", "full"):
        raise ValueError(f"covariance_type must be 'diag' or 'full', got '{covariance_type}'")

    emp_var = np.var(X, axis=0) + reg
    if covariance_type == "full":
        emp_cov = (np.cov(X.T, ddof=0) + reg * np.eye(d)) if d > 1 else emp_var.reshape(1, 1)

    best_model = None

    for restart in range(n_init):
        # ── k-means++ initialisation ─────────────────────────────────────────
        rng  = np.random.default_rng(random_state + restart)
        idx0 = rng.integers(n)
        centers = [X[idx0]]
        for _ in range(k - 1):
            # Vectorised min-distance to existing centers
            dists = np.min(
                np.sum((X[:, np.newaxis, :] - np.array(centers)[np.newaxis, :, :]) ** 2, axis=2),
                axis=1,
            )
            probs = dists / (dists.sum() + _EPS)
            centers.append(X[rng.choice(n, p=probs)])
        means = np.array(centers)

        # ── initial covariances ───────────────────────────────────────────────
        if covariance_type == "diag":
            covs = np.tile(emp_var, (k, 1))
        else:
            covs = np.tile(emp_cov[np.newaxis], (k, 1, 1))

        weights  = np.full(k, 1.0 / k)
        prev_lik = -np.inf
        avg_lik  = -np.inf

        for iteration in range(1, max_iter + 1):
            log_resp, log_liks = _e_step(X, weights, means, covs, covariance_type)
            avg_lik = float(log_liks.mean())

            if abs(avg_lik - prev_lik) < tol:
                break

            prev_lik             = avg_lik
            weights, means, covs = _m_step(X, log_resp, covariance_type, reg=reg)

        model = {
            "weights"  : weights,
            "means"    : means,
            "covs"     : covs,
            "cov_type" : covariance_type,
            "log_lik"  : avg_lik,
            "n_iter"   : iteration,
        }
        if best_model is None or avg_lik > best_model["log_lik"]:
            best_model = model

    return best_model


def gmm_log_prob(X: np.ndarray, model: dict) -> np.ndarray:
    """
    Per-sample log p(x) under the fitted GMM.

    Returns
    -------
    log_probs : (n_samples,)
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    _, log_liks = _e_step(
        X, model["weights"], model["means"], model["covs"], model["cov_type"]
    )
    return log_liks


def gmm_predict(X: np.ndarray, model: dict) -> np.ndarray:
    """
    Hard cluster assignment: argmax_k r_nk for each sample.

    Returns
    -------
    labels : (n_samples,) int array
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    log_resp, _ = _e_step(
        X, model["weights"], model["means"], model["covs"], model["cov_type"]
    )
    return np.argmax(log_resp, axis=1)


def gmm_bic(X: np.ndarray, model: dict) -> float:
    """
    Bayesian Information Criterion.  Lower = better model.

    BIC = −2·ℓ + p·log(N)
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, np.newaxis]

    n, d      = X.shape
    k         = len(model["weights"])
    total_lik = gmm_log_prob(X, model).sum()

    if model["cov_type"] == "diag":
        n_params = k * (2 * d + 1) - 1
    else:
        n_params = k * (d + d * (d + 1) // 2) + (k - 1)

    return -2.0 * total_lik + n_params * np.log(n)


def select_n_components(
    X: np.ndarray,
    n_range,
    covariance_type: str = "diag",
    **fit_kwargs,
) -> tuple[int, dict, list]:
    """
    Fit GMMs for each k in n_range; return the one with lowest BIC.

    Parameters
    ----------
    X            : (n_samples, n_features)
    n_range      : iterable of candidate k values, e.g. range(1, 7)
    covariance_type : 'diag' or 'full'
    **fit_kwargs : forwarded to gmm_fit (max_iter, tol, reg, random_state)

    Returns
    -------
    best_k     : int   — k with the lowest BIC
    best_model : dict  — fitted GMM for best_k
    bics       : list of (k, bic) pairs in n_range order
    """
    X          = np.asarray(X, dtype=np.float64)
    bics       = []
    best_bic   = np.inf
    best_k     = None
    best_model = None

    for k in n_range:
        model = gmm_fit(X, k, covariance_type=covariance_type, **fit_kwargs)
        b     = gmm_bic(X, model)
        bics.append((k, b))
        if b < best_bic:
            best_bic   = b
            best_k     = k
            best_model = model

    return best_k, best_model, bics
