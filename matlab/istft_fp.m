function y = istft_fp(S, sample_rate, frame_len, hop_len, original_len)
%ISTFT_FP Inverse STFT (overlap-add) matching `audio_filter/stft.py`.
%
% Inputs:
%   S            (n_frames x n_bins) complex one-sided spectrum
%   sample_rate  unused (kept for API symmetry)
%   frame_len    samples per frame
%   hop_len      samples per hop
%   original_len if provided, trim output to this many samples

%#ok<NASGU>

n_frames   = size(S, 1);
out_len    = (n_frames - 1) * hop_len + frame_len;
center_pad = floor(frame_len / 2);

w = hann_window_periodic(frame_len);

y = zeros(out_len, 1);
window_sum = zeros(out_len, 1);

for i = 1:n_frames
    start = (i - 1) * hop_len + 1;
    frame = local_irfft(S(i, :), frame_len);

    y(start:start+frame_len-1) = y(start:start+frame_len-1) + frame(:) .* w;
    window_sum(start:start+frame_len-1) = window_sum(start:start+frame_len-1) + (w .^ 2);
end

nz = window_sum > 1e-8;
y(nz) = y(nz) ./ window_sum(nz);

% Strip center padding introduced by stft_fp()
y = y(center_pad+1:end);

if nargin >= 5 && ~isempty(original_len)
    y = y(1:original_len);
end
end

function frame = local_irfft(Srow, n)
%LOCAL_IRFFT Real IFFT from one-sided spectrum (rfft-style).
% Srow: 1 x (n//2+1)

Srow = reshape(Srow, 1, []);

if mod(n, 2) == 0
    tail = conj(Srow(end-1:-1:2));
else
    tail = conj(Srow(end:-1:2));
end

X = [Srow, tail];
frame = real(ifft(X, n, 2));
end
