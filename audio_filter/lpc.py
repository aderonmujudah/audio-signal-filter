"""
LPC + formant extractor — built from first principles using NumPy.

Pipeline per frame:
  [pre-emphasis] → window → autocorrelation → Levinson-Durbin → LPC coefficients
                                                                     ↓
                                               np.roots → filter poles
                                                             ↓
                                               angle → frequency, |r| → bandwidth
                                                             ↓
                                               sort + filter → F1, F2, F3 ...

Math summary:
  LPC order p: model speech as AR(p) filter,  s[n] = -sum_{k=1}^p a[k] s[n-k] + e[n]

  Yule-Walker:  [R(0)  R(1) ... R(p-1)] [a[1]]   [-R(1)]
                [R(1)  R(0) ... R(p-2)] [a[2]] = [-R(2)]
                [      ...            ] [ ... ]   [ ... ]
                [R(p-1) ...     R(0)  ] [a[p]]   [-R(p)]

  Solved via Levinson-Durbin (O(p²)).

  Formant from pole r:
    frequency  f  = angle(r) * sr / (2π)     [Hz]
    bandwidth  bw = -ln(|r|) * sr / π        [Hz]
"""

import numpy as np
from .stft import hann_window


# ---------------------------------------------------------------------------
# Pre-emphasis
# ---------------------------------------------------------------------------

def pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    First-order high-pass filter: y[n] = x[n] - coeff * x[n-1].
    Flattens the speech spectrum before LPC analysis.
    """
    return np.concatenate([[signal[0]], signal[1:] - coeff * signal[:-1]])


# ---------------------------------------------------------------------------
# Internal framing  (mirrors stft.py center-pad convention)
# ---------------------------------------------------------------------------

def _make_frames(
    signal: np.ndarray,
    frame_len: int,
    hop_len: int,
    apply_window: bool = True,
) -> np.ndarray:
    """Return windowed frames, shape (n_frames, frame_len)."""
    center_pad = frame_len // 2
    sig = np.concatenate([np.zeros(center_pad), signal, np.zeros(center_pad)])

    n_frames  = 1 + (len(sig) - frame_len + hop_len - 1) // hop_len
    pad_right = (n_frames - 1) * hop_len + frame_len - len(sig)
    if pad_right > 0:
        sig = np.concatenate([sig, np.zeros(pad_right)])

    indices = (
        np.arange(frame_len)[np.newaxis, :]
        + np.arange(n_frames)[:, np.newaxis] * hop_len
    )
    frames = sig[indices]   # (n_frames, frame_len)

    if apply_window:
        frames = frames * hann_window(frame_len)

    return frames


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def autocorrelation(frame: np.ndarray, order: int) -> np.ndarray:
    """
    Biased autocorrelation R[0..order] of a single frame.

    R[k] = (1/N) * sum_{n=0}^{N-1-k} frame[n] * frame[n+k]
    """
    N = len(frame)
    R = np.empty(order + 1)
    for k in range(order + 1):
        R[k] = np.dot(frame[:N - k], frame[k:]) / N
    return R


# ---------------------------------------------------------------------------
# Levinson-Durbin recursion
# ---------------------------------------------------------------------------

def levinson_durbin(R: np.ndarray) -> np.ndarray:
    """
    Solve the Yule-Walker equations via Levinson-Durbin recursion.

    Parameters
    ----------
    R : 1-D array, autocorrelation lags R[0..p]  (length = p+1)

    Returns
    -------
    a : 1-D array of LPC coefficients [a_1, ..., a_p]
        The prediction error filter is A(z) = 1 + a_1 z^{-1} + ... + a_p z^{-p}
    """
    order = len(R) - 1

    if R[0] < 1e-10:          # silent frame
        return np.zeros(order)

    a = np.zeros(order)       # a[i] stores coefficient a_{i+1} in math notation
    E = float(R[0])

    for i in range(order):
        # Reflection coefficient  k_{i+1}
        k = -(R[i + 1] + np.dot(a[:i], R[i:0:-1])) / E

        # Update coefficients in-place (must use a copy of old values)
        a_prev = a[:i].copy()
        a[i]   = k
        a[:i] += k * a_prev[::-1]

        E *= (1.0 - k * k)

        if E <= 1e-10:          # numerical blow-up guard
            break

    return a


# ---------------------------------------------------------------------------
# LPC per frame
# ---------------------------------------------------------------------------

def lpc(
    signal: np.ndarray,
    sample_rate: int,
    order: int | None = None,
    frame_sec: float = 0.025,
    hop_sec: float = 0.010,
    emphasis: float = 0.97,
) -> tuple[np.ndarray, int, int]:
    """
    Compute LPC coefficients for every frame of a mono signal.

    Parameters
    ----------
    signal      : 1-D float array
    sample_rate : samples per second
    order       : LPC order p  (default: 2 + sample_rate // 1000)
    frame_sec   : frame length in seconds
    hop_sec     : hop size in seconds
    emphasis    : pre-emphasis coefficient (0 = disabled)

    Returns
    -------
    A         : float array, shape (n_frames, order)
                Row i is [a_1, ..., a_p] for frame i
    frame_len : int
    hop_len   : int
    """
    signal = np.asarray(signal, dtype=np.float64)

    if order is None:
        order = 2 + sample_rate // 1000     # e.g. 18 for 16 kHz

    frame_len = int(round(frame_sec * sample_rate))
    hop_len   = int(round(hop_sec   * sample_rate))

    if emphasis > 0:
        signal = pre_emphasis(signal, emphasis)

    frames = _make_frames(signal, frame_len, hop_len)   # (n_frames, frame_len)
    n_frames = frames.shape[0]

    A = np.zeros((n_frames, order))
    for i in range(n_frames):
        R    = autocorrelation(frames[i], order)
        A[i] = levinson_durbin(R)

    return A, frame_len, hop_len


# ---------------------------------------------------------------------------
# Formant extraction from LPC coefficients
# ---------------------------------------------------------------------------

def lpc_to_formants(
    A: np.ndarray,
    sample_rate: int,
    n_formants: int = 4,
    min_freq: float = 90.0,
    max_bw: float = 400.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract formant frequencies and bandwidths from per-frame LPC coefficients.

    For each frame:
      1. Find roots of the LPC polynomial [1, a_1, ..., a_p].
      2. Convert roots to frequency/bandwidth.
      3. Keep the lowest-frequency roots that pass the voiced-speech filter.

    Parameters
    ----------
    A          : float array, shape (n_frames, order)
    sample_rate: samples per second
    n_formants : how many formants to return per frame  (default 4)
    min_freq   : minimum formant frequency in Hz       (default 90 Hz)
    max_bw     : maximum formant bandwidth in Hz       (default 400 Hz)

    Returns
    -------
    freqs : float array, shape (n_frames, n_formants)  — Hz, NaN if not found
    bws   : float array, shape (n_frames, n_formants)  — Hz, NaN if not found
    """
    n_frames = A.shape[0]
    freqs    = np.full((n_frames, n_formants), np.nan)
    bws      = np.full((n_frames, n_formants), np.nan)

    for i in range(n_frames):
        poly  = np.concatenate([[1.0], A[i]])   # LPC polynomial coefficients
        roots = np.roots(poly)

        # Keep roots with positive imaginary part (one from each conjugate pair)
        roots = roots[roots.imag >= 0]

        if len(roots) == 0:
            continue

        # Convert to frequency and bandwidth
        magnitudes = np.abs(roots)
        angles     = np.angle(roots)
        f          = angles * sample_rate / (2.0 * np.pi)
        # Guard against |r| == 0 before log (zero-magnitude root → infinite bw, filtered out)
        bw         = -np.log(np.maximum(magnitudes, 1e-15)) * sample_rate / np.pi

        # Voiced-speech filter: positive frequency, not too narrow/wide
        mask = (f > min_freq) & (bw > 0) & (bw < max_bw)
        f, bw = f[mask], bw[mask]

        if len(f) == 0:
            continue

        # Sort by frequency, fill up to n_formants slots
        order = np.argsort(f)
        f, bw = f[order], bw[order]
        n = min(len(f), n_formants)
        freqs[i, :n] = f[:n]
        bws[i,   :n] = bw[:n]

    return freqs, bws


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def formants(
    signal: np.ndarray,
    sample_rate: int,
    order: int | None = None,
    n_formants: int = 4,
    frame_sec: float = 0.025,
    hop_sec: float = 0.010,
    emphasis: float = 0.97,
) -> tuple[np.ndarray, np.ndarray]:
    """
    End-to-end: signal → per-frame formant frequencies and bandwidths.

    Returns
    -------
    freqs : (n_frames, n_formants) Hz
    bws   : (n_frames, n_formants) Hz
    """
    A, _, _ = lpc(signal, sample_rate, order, frame_sec, hop_sec, emphasis)
    return lpc_to_formants(A, sample_rate, n_formants)
