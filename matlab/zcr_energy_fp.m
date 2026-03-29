function [zcr_track, energy_track] = zcr_energy_fp(signal, sample_rate, frame_sec, hop_sec)
%ZCR_ENERGY_FP Per-frame ZCR and mean-squared energy.
% Mirrors `audio_filter/zcr_energy.py`.
%
% Returns:
%   zcr_track    (n_frames x 1)  in [0, 1]
%   energy_track (n_frames x 1)  >= 0

if nargin < 3 || isempty(frame_sec),  frame_sec = 0.025; end
if nargin < 4 || isempty(hop_sec),    hop_sec   = 0.010; end

x         = double(signal(:));
frame_len = round(frame_sec * sample_rate);
hop_len   = round(hop_sec   * sample_rate);

frames   = local_make_frames(x, frame_len, hop_len);  % (n_frames x frame_len)
n_frames = size(frames, 1);

% ZCR: count sign changes between consecutive samples
sign_prod   = frames(:, 2:end) .* frames(:, 1:end-1);
crossings   = sum(sign_prod < 0, 2);
zcr_track   = crossings / (frame_len - 1);

% Mean-squared energy
energy_track = mean(frames .^ 2, 2);
end


function frames = local_make_frames(x, frame_len, hop_len)
center_pad = floor(frame_len / 2);
x = [zeros(center_pad, 1); x; zeros(center_pad, 1)];

n_frames  = 1 + floor((length(x) - frame_len + hop_len - 1) / hop_len);
pad_right = (n_frames - 1) * hop_len + frame_len - length(x);
if pad_right > 0
    x = [x; zeros(pad_right, 1)];
end

idx    = (0:frame_len-1)' + (0:n_frames-1) * hop_len;
frames = x(idx + 1).';   % (n_frames x frame_len) — no Hann window
end
