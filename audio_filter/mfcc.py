"""
MFCC extractor — built from first principles using NumPy.

Pipeline per frame:
  rfft → power spectrum → mel filterbank → log → DCT-II  →  n_mfcc coefficients

Mel filterbank math:
  mel  = 2595 * log10(1 + f / 700)        (Hz → mel)
  f    = 700  * (10^(mel / 2595) - 1)     (mel → Hz)
  Triangular filters equally spaced on the mel scale,
  mapped back to FFT bin indices.

DCT-II (orthonormal):
  C[k] = sqrt(2/M) * sum_{m=0}^{M-1} x[m] * cos(π k (m + 0.5) / M)
  with the k=0 row scaled by 1/sqrt(2) for orthonormality.
"""

import numpy as np
from .stft import stft


# ---------------------------------------------------------------------------
# Mel ↔ Hz conversion
# ---------------------------------------------------------------------------

def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


# ---------------------------------------------------------------------------
# Mel filterbank
# ---------------------------------------------------------------------------

def mel_filterbank(
    frame_len: int,
    sample_rate: int,
    n_mels: int = 26,
    fmin: float = 80.0,
    fmax: float | None = None,
) -> np.ndarray:
    """
    Build a mel filterbank matrix.

    Parameters
    ----------
    frame_len   : FFT size (= samples per STFT frame)
    sample_rate : samples per second
    n_mels      : number of mel filters  (default 26)
    fmin        : lowest filter frequency in Hz  (default 80 Hz)
    fmax        : highest filter frequency in Hz (default Nyquist)

    Returns
    -------
    fb : float array, shape (n_mels, frame_len // 2 + 1)
         Each row is one triangular filter over FFT bins.
    """
    if fmax is None:
        fmax = sample_rate / 2.0

    n_bins = frame_len // 2 + 1

    # n_mels + 2 equally spaced mel points including the two boundary points
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)

    # Map Hz to nearest FFT bin  (standard formula)
    bins = np.floor((frame_len + 1) * hz_pts / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)

    fb  = np.zeros((n_mels, n_bins), dtype=np.float64)
    k   = np.arange(n_bins, dtype=np.float64)    # bin index vector

    for m in range(n_mels):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]

        # Rising slope  [left, center)
        if center > left:
            mask = (k >= left) & (k < center)
            fb[m, mask] = (k[mask] - left) / (center - left)

        # Falling slope  [center, right]
        if right > center:
            mask = (k >= center) & (k <= right)
            fb[m, mask] = (right - k[mask]) / (right - center)

    return fb


# ---------------------------------------------------------------------------
# DCT-II (orthonormal)
# ---------------------------------------------------------------------------

def dct_matrix(n_input: int, n_output: int) -> np.ndarray:
    """
    Orthonormal DCT-II matrix of shape (n_output, n_input).

    Applying this to a row vector x gives the DCT coefficients C:
        C = x @ D.T     (or equivalently  D @ x  for a column vector)
    """
    m = np.arange(n_input,  dtype=np.float64)[np.newaxis, :]  # (1, n_input)
    k = np.arange(n_output, dtype=np.float64)[:, np.newaxis]  # (n_output, 1)

    D = np.cos(np.pi * k * (m + 0.5) / n_input)   # (n_output, n_input)

    # Orthonormal scaling
    D[0]  *= 1.0 / np.sqrt(n_input)
    D[1:] *= np.sqrt(2.0 / n_input)

    return D


# ---------------------------------------------------------------------------
# MFCC
# ---------------------------------------------------------------------------

def mfcc(
    signal: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_mels: int = 26,
    frame_sec: float = 0.025,
    hop_sec: float = 0.010,
    fmin: float = 80.0,
    fmax: float | None = None,
) -> np.ndarray:
    """
    Compute MFCCs for a mono audio signal.

    Parameters
    ----------
    signal      : 1-D float array
    sample_rate : samples per second
    n_mfcc      : number of cepstral coefficients to return  (default 13)
    n_mels      : number of mel filters  (default 26)
    frame_sec   : frame length in seconds  (default 25 ms)
    hop_sec     : hop size in seconds      (default 10 ms)
    fmin        : lowest mel filter frequency in Hz
    fmax        : highest mel filter frequency in Hz  (default Nyquist)

    Returns
    -------
    coeffs : float array, shape (n_frames, n_mfcc)
    """
    # 1. STFT → power spectrum
    S, frame_len, _ = stft(signal, sample_rate, frame_sec, hop_sec)
    power = np.abs(S) ** 2                             # (n_frames, n_bins)

    # 2. Mel filterbank
    fb = mel_filterbank(frame_len, sample_rate, n_mels, fmin, fmax)

    # 3. Apply filters → log mel energies
    mel_energy = power @ fb.T                          # (n_frames, n_mels)
    log_mel    = np.log(np.maximum(mel_energy, 1e-10)) # floor avoids log(0)

    # 4. DCT-II → cepstral coefficients
    D      = dct_matrix(n_mels, n_mfcc)               # (n_mfcc, n_mels)
    coeffs = log_mel @ D.T                             # (n_frames, n_mfcc)

    return coeffs


# ---------------------------------------------------------------------------
# Delta features  (first-order and second-order time derivatives)
# ---------------------------------------------------------------------------

def delta(features: np.ndarray, width: int = 9) -> np.ndarray:
    """
    Compute delta (first derivative) of a feature matrix using a
    linear regression over a symmetric context window.

    delta[t] = sum_{d=1}^{N} d * (feat[t+d] - feat[t-d])
               / (2 * sum_{d=1}^{N} d^2)
    where N = (width - 1) // 2.

    Parameters
    ----------
    features : (n_frames, n_features)
    width    : context window width, must be odd  (default 9)

    Returns
    -------
    deltas : (n_frames, n_features)
    """
    assert width % 2 == 1 and width >= 3, "width must be odd and >= 3"
    N = (width - 1) // 2

    # Pad edges by replication so all frames have a full context window
    padded = np.pad(features, ((N, N), (0, 0)), mode="edge")

    denominator = 2.0 * np.sum(np.arange(1, N + 1) ** 2)
    result      = np.zeros_like(features)

    for d in range(1, N + 1):
        result += d * (padded[N + d : N + d + len(features)]
                      - padded[N - d : N - d + len(features)])

    return result / denominator
