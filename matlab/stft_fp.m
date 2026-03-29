function [S, frame_len, hop_len] = stft_fp(signal, sample_rate, frame_sec, hop_sec)
%STFT_FP Short-Time Fourier Transform from first principles.
%
% Mirrors `audio_filter/stft.py`:
% - center padding by frame_len/2
% - Hann window (periodic)
% - ceil-divide framing + right padding
% - one-sided FFT bins (rfft-style)
%
% Returns:
%   S         (n_frames x n_bins) complex
%   frame_len samples per frame
%   hop_len   samples per hop

if nargin < 3 || isempty(frame_sec)
    frame_sec = 0.025;
end
if nargin < 4 || isempty(hop_sec)
    hop_sec = 0.010;
end

x = double(signal(:));
frame_len  = round(frame_sec * sample_rate);
hop_len    = round(hop_sec * sample_rate);
center_pad = floor(frame_len / 2);

x = [zeros(center_pad, 1); x; zeros(center_pad, 1)];

n_frames  = 1 + floor((length(x) - frame_len + hop_len - 1) / hop_len);
pad_right = (n_frames - 1) * hop_len + frame_len - length(x);
if pad_right > 0
    x = [x; zeros(pad_right, 1)];
end

w = hann_window_periodic(frame_len);

% Build frame matrix: frames is (n_frames x frame_len)
idx = (0:frame_len-1)' + (0:n_frames-1) * hop_len;   % (frame_len x n_frames)
frames = (x(idx + 1) .* w).';

Sfull = fft(frames, frame_len, 2);
nbins = floor(frame_len/2) + 1;
S = Sfull(:, 1:nbins);
end
