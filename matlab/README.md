# MATLAB reference implementations

These `.m` files mirror the Python implementations in `audio_filter/`.

## Files

- `stft_fp.m`, `istft_fp.m` — STFT/ISTFT with center-padding + overlap-add normalization
- `mfcc_fp.m` — MFCCs using mel filterbank + orthonormal DCT-II
- `lpc_framewise_fp.m` — LPC via autocorrelation + Levinson–Durbin
- `lpc_to_formants_fp.m` — Formant frequencies/bandwidths via LPC roots

## Run

```matlab
addpath('matlab');
run('matlab/run_smoke_tests.m');
```

## Naming note

MATLAB ships built-ins like `stft`; these files use the `_fp` suffix to avoid collisions.
