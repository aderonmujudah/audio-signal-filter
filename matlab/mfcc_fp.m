function coeffs = mfcc_fp(signal, sample_rate, n_mfcc, n_mels, frame_sec, hop_sec, fmin, fmax)
%MFCC_FP MFCC extractor from first principles (matches `audio_filter/mfcc.py`).
%
% Returns coeffs of shape (n_frames x n_mfcc)

if nargin < 3 || isempty(n_mfcc)
    n_mfcc = 13;
end
if nargin < 4 || isempty(n_mels)
    n_mels = 26;
end
if nargin < 5 || isempty(frame_sec)
    frame_sec = 0.025;
end
if nargin < 6 || isempty(hop_sec)
    hop_sec = 0.010;
end
if nargin < 7 || isempty(fmin)
    fmin = 80.0;
end
if nargin < 8 || isempty(fmax)
    fmax = sample_rate / 2.0;
end

[S, frame_len, ~] = stft_fp(signal, sample_rate, frame_sec, hop_sec);
power = abs(S) .^ 2;

fb = mel_filterbank_fp(frame_len, sample_rate, n_mels, fmin, fmax);

mel_energy = power * fb.';
log_mel = log(max(mel_energy, 1e-10));

D = dct_matrix_fp(n_mels, n_mfcc);
coeffs = log_mel * D.';
end
