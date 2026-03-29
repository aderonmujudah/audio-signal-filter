function fb = mel_filterbank_fp(frame_len, sample_rate, n_mels, fmin, fmax)
%MEL_FILTERBANK_FP Mel triangular filterbank (matches `audio_filter/mfcc.py`).
%
% Returns fb of shape (n_mels x (frame_len/2+1))

if nargin < 3 || isempty(n_mels)
    n_mels = 26;
end
if nargin < 4 || isempty(fmin)
    fmin = 80.0;
end
if nargin < 5 || isempty(fmax)
    fmax = sample_rate / 2.0;
end

n_bins = floor(frame_len/2) + 1;

mel_pts = linspace(hz_to_mel_fp(fmin), hz_to_mel_fp(fmax), n_mels + 2);
hz_pts  = mel_to_hz_fp(mel_pts);

bins = floor((frame_len + 1) * hz_pts / sample_rate);
bins = max(0, min(n_bins - 1, bins));

fb = zeros(n_mels, n_bins);
k = 0:(n_bins - 1);

for m = 1:n_mels
    left   = bins(m);
    center = bins(m + 1);
    right  = bins(m + 2);

    if center > left
        mask = (k >= left) & (k < center);
        fb(m, mask) = (k(mask) - left) ./ (center - left);
    end

    if right > center
        mask = (k >= center) & (k <= right);
        fb(m, mask) = (right - k(mask)) ./ (right - center);
    end
end
end
