"""
STFT / ISTFT engine — built from first principles using NumPy.

Short-Time Fourier Transform pipeline:
  signal → [center-pad] → frame → window → rfft  =>  complex spectrogram
  complex spectrogram → irfft → window → overlap-add → [trim] → signal

Center-padding by frame_len//2 at both ends ensures every original sample
lands in the interior of at least one frame, avoiding near-zero Hann-window
edge values that would blow up the OLA normalization.

Typical speech parameters:
  frame_sec : 0.025 s  (25 ms)
  hop_sec   : 0.010 s  (10 ms, 60 % overlap)
  window    : Hann
"""

import numpy as np


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

def hann_window(n: int) -> np.ndarray:
    """Periodic Hann window of length n.  w[k] = 0.5 * (1 - cos(2π k / n))"""
    k = np.arange(n)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * k / n))


# ---------------------------------------------------------------------------
# STFT
# ---------------------------------------------------------------------------

def stft(
    signal: np.ndarray,
    sample_rate: int,
    frame_sec: float = 0.025,
    hop_sec: float = 0.010,
) -> tuple[np.ndarray, int, int]:
    """
    Compute the Short-Time Fourier Transform.

    Parameters
    ----------
    signal      : 1-D float array, mono audio samples
    sample_rate : samples per second
    frame_sec   : frame length in seconds  (default 25 ms)
    hop_sec     : hop size in seconds      (default 10 ms)

    Returns
    -------
    S           : complex array, shape (n_frames, n_bins)
                  n_bins = frame_len // 2 + 1  (one-sided spectrum)
    frame_len   : int, samples per frame
    hop_len     : int, samples per hop
    """
    signal = np.asarray(signal, dtype=np.float64)

    frame_len  = int(round(frame_sec * sample_rate))
    hop_len    = int(round(hop_sec   * sample_rate))
    center_pad = frame_len // 2

    # Center-pad: shift every original sample away from the Hann-window zeros
    # at the frame edges.  ISTFT must strip the same center_pad from the front.
    signal = np.concatenate([np.zeros(center_pad), signal, np.zeros(center_pad)])

    # Ceil-divide so the last sample is covered by at least one frame
    n_frames = 1 + (len(signal) - frame_len + hop_len - 1) // hop_len
    pad_right = (n_frames - 1) * hop_len + frame_len - len(signal)
    if pad_right > 0:
        signal = np.concatenate([signal, np.zeros(pad_right)])

    window = hann_window(frame_len)

    # Build frame matrix  (n_frames, frame_len)
    indices = (
        np.arange(frame_len)[np.newaxis, :]             # (1, frame_len)
        + np.arange(n_frames)[:, np.newaxis] * hop_len  # (n_frames, 1)
    )
    frames = signal[indices] * window    # apply Hann window to each frame

    # rfft along the frame axis → (n_frames, n_bins)
    S = np.fft.rfft(frames, n=frame_len, axis=1)

    return S, frame_len, hop_len


# ---------------------------------------------------------------------------
# ISTFT
# ---------------------------------------------------------------------------

def istft(
    S: np.ndarray,
    sample_rate: int,
    frame_len: int,
    hop_len: int,
    original_len: int | None = None,
) -> np.ndarray:
    """
    Reconstruct a time-domain signal from a complex STFT.

    Parameters
    ----------
    S            : complex array, shape (n_frames, n_bins)
    sample_rate  : samples per second (kept for API symmetry)
    frame_len    : samples per frame — must match the stft() call
    hop_len      : samples per hop   — must match the stft() call
    original_len : if given, trim the output to this many samples

    Returns
    -------
    signal : 1-D float array
    """
    n_frames   = S.shape[0]
    out_len    = (n_frames - 1) * hop_len + frame_len
    center_pad = frame_len // 2

    window     = hann_window(frame_len)
    signal     = np.zeros(out_len, dtype=np.float64)
    window_sum = np.zeros(out_len, dtype=np.float64)

    # irfft each frame → (n_frames, frame_len)
    frames = np.fft.irfft(S, n=frame_len, axis=1)

    for i in range(n_frames):
        start = i * hop_len
        signal[start : start + frame_len]     += frames[i] * window
        window_sum[start : start + frame_len] += window ** 2

    # Normalize (every interior sample has a nonzero window sum)
    nz = window_sum > 1e-8
    signal[nz] /= window_sum[nz]

    # Strip the center-pad introduced by stft()
    signal = signal[center_pad:]

    if original_len is not None:
        signal = signal[:original_len]

    return signal


# ---------------------------------------------------------------------------
# Convenience: magnitude & phase
# ---------------------------------------------------------------------------

def magnitude_phase(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split complex STFT into magnitude and phase."""
    return np.abs(S), np.angle(S)


def reconstruct_from_magnitude(
    magnitude: np.ndarray,
    phase: np.ndarray,
) -> np.ndarray:
    """Combine magnitude + phase back to a complex STFT."""
    return magnitude * np.exp(1j * phase)
