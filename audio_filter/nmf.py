"""
NMF (Non-negative Matrix Factorization) + Griffin-Lim phase reconstruction.

NMF — multiplicative update rules (Euclidean/Frobenius, Lee & Seung 2001):
    Factorize  V ≈ W · H,  V, W, H ≥ 0

    Update rules (guaranteed non-increasing reconstruction error):
        H  ←  H  ⊙  (Wᵀ V)     ⊘  (Wᵀ W H  + ε)
        W  ←  W  ⊙  (V Hᵀ)     ⊘  (W H Hᵀ  + ε)

    where ⊙ / ⊘ are element-wise multiply / divide and ε = 1e-10.

Convention: V is (n_bins, n_frames).  The STFT module returns S as
(n_frames, n_bins), so transpose before calling nmf:

    S, fl, hl = stft(signal, sr)
    V = np.abs(S).T                  # (n_bins, n_frames)
    W, H = nmf(V, n_components=2)
    mag_src = W[:, 0:1] * H[0:1, :] # (n_bins, n_frames)  source 0
    x_src   = griffin_lim(mag_src, fl, hl, sr)

Griffin-Lim (Griffin & Lim 1984):
    Estimate a time-domain signal from a magnitude-only spectrogram by
    iterating between the space of STFT-consistent spectrograms and the
    manifold of spectrograms with the target magnitude.

    Algorithm:
        Φ ← random initial phases
        for n in range(n_iter):
            S  = magnitude ⊙ exp(jΦ)
            x  = ISTFT(S)
            S' = STFT(x)
            Φ  = angle(S')
        return x
"""

import numpy as np

from .stft import stft as _stft, istft as _istft

_EPS = 1e-10


# ---------------------------------------------------------------------------
# NMF
# ---------------------------------------------------------------------------

def nmf(
    V: np.ndarray,
    n_components: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Factorize non-negative matrix V ≈ W · H (Euclidean multiplicative updates).

    Parameters
    ----------
    V            : (n_bins, n_frames)  non-negative input matrix
    n_components : number of components / sources
    max_iter     : maximum multiplicative update iterations
    tol          : early-stop when relative change in ||V − WH||_F < tol
    random_state : seed for reproducible initialisation

    Returns
    -------
    W : (n_bins, n_components)  basis matrix (spectral shapes)
    H : (n_components, n_frames) activation matrix (temporal envelopes)
    """
    V = np.asarray(V, dtype=np.float64)
    if V.ndim != 2:
        raise ValueError(f"V must be 2-D, got shape {V.shape}")
    if np.any(V < 0):
        raise ValueError("V must be non-negative")

    n_bins, n_frames = V.shape
    rng = np.random.default_rng(random_state)

    W = rng.random((n_bins, n_components)) + _EPS
    H = rng.random((n_components, n_frames)) + _EPS

    prev_err = None

    for _ in range(max_iter):
        # ── update H ─────────────────────────────────────────────────────────
        WH  = W @ H
        H  *= (W.T @ V) / (W.T @ WH + _EPS)

        # ── update W ─────────────────────────────────────────────────────────
        WH  = W @ H
        W  *= (V @ H.T) / (WH @ H.T + _EPS)

        # ── convergence check ─────────────────────────────────────────────────
        WH  = W @ H
        err = np.linalg.norm(V - WH, "fro")
        if prev_err is not None and abs(prev_err - err) / (prev_err + _EPS) < tol:
            break
        prev_err = err

    return W, H


# ---------------------------------------------------------------------------
# Griffin-Lim
# ---------------------------------------------------------------------------

def griffin_lim(
    magnitude: np.ndarray,
    frame_len: int,
    hop_len: int,
    sample_rate: int,
    n_iter: int = 32,
    random_state: int = 0,
) -> np.ndarray:
    """
    Reconstruct a time-domain signal from a magnitude spectrogram.

    Parameters
    ----------
    magnitude   : (n_bins, n_frames)  non-negative magnitude spectrogram.
                  Use np.abs(S).T to convert from stft() output (n_frames, n_bins).
    frame_len   : STFT frame length in samples — must match the analysis stft() call
    hop_len     : STFT hop length in samples   — must match the analysis stft() call
    sample_rate : sample rate in Hz
    n_iter      : Griffin-Lim iterations (32 is usually sufficient)
    random_state: seed for initial random phase

    Returns
    -------
    signal : (n_samples,) reconstructed time-domain signal
    """
    magnitude = np.asarray(magnitude, dtype=np.float64)
    if magnitude.ndim != 2:
        raise ValueError(f"magnitude must be 2-D (n_bins, n_frames), got {magnitude.shape}")

    n_bins, n_frames = magnitude.shape
    frame_sec = frame_len / sample_rate
    hop_sec   = hop_len   / sample_rate

    # Signal length that reproduces exactly n_frames when passed through stft().
    # stft() pads by frame_len//2 on each side, so:
    #   n_frames = 1 + ceil(target_len / hop_len)
    #   target_len = (n_frames - 1) * hop_len  satisfies ceil(target/hop) = n_frames-1
    target_len = (n_frames - 1) * hop_len

    rng   = np.random.default_rng(random_state)
    phase = rng.uniform(-np.pi, np.pi, (n_bins, n_frames))

    signal = np.zeros(target_len)   # initial fallback

    for _ in range(n_iter):
        # Build complex spectrogram  (n_bins, n_frames)  → transpose for ISTFT
        S_complex = magnitude * np.exp(1j * phase)       # (n_bins, n_frames)

        # ISTFT expects (n_frames, n_bins); trim to target_len keeps frame count stable
        signal = _istft(S_complex.T, sample_rate, frame_len, hop_len,
                        original_len=target_len)

        # Re-analyse
        S_new, _, _ = _stft(signal, sample_rate,
                            frame_sec=frame_sec,
                            hop_sec=hop_sec)              # (n_frames_new, n_bins)

        S_new = S_new.T                                   # → (n_bins, n_frames_new)

        # Frame counts should match; guard against off-by-one
        n_new = S_new.shape[1]
        if n_new >= n_frames:
            phase = np.angle(S_new[:, :n_frames])
        else:
            phase = np.pad(np.angle(S_new), ((0, 0), (0, n_frames - n_new)))

    return signal
