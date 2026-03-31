function [W, H] = nmf_fp(V, n_components, max_iter, tol, random_state)
%NMF_FP  Non-negative Matrix Factorization (multiplicative updates, Frobenius).
% Mirrors `audio_filter/nmf.py`.
%
% Factorizes V ≈ W * H  using the Lee & Seung multiplicative update rules
% (Euclidean / Frobenius norm).
%
%   H  ←  H  .* (W' * V)     ./ (W' * W * H  + eps)
%   W  ←  W  .* (V  * H')    ./ (W  * H * H' + eps)
%
% V            : (n_bins x n_frames)  non-negative input matrix
% n_components : number of components
%
% Returns:
%   W : (n_bins x n_components)
%   H : (n_components x n_frames)

if nargin < 3 || isempty(max_iter),     max_iter     = 300;   end
if nargin < 4 || isempty(tol),          tol          = 1e-4;  end
if nargin < 5 || isempty(random_state), random_state = 0;     end

V   = double(V);
EPS = 1e-10;

assert(all(V(:) >= 0), 'V must be non-negative');

[n_bins, n_frames] = size(V);

rng(random_state);
W = rand(n_bins, n_components) + EPS;
H = rand(n_components, n_frames) + EPS;

prev_err = inf;

for iter = 1:max_iter
    % ── update H ─────────────────────────────────────────────────────────────
    WH = W * H;
    H  = H .* ((W' * V) ./ (W' * WH + EPS));

    % ── update W ─────────────────────────────────────────────────────────────
    WH = W * H;
    W  = W .* ((V * H') ./ (WH * H' + EPS));

    % ── convergence ──────────────────────────────────────────────────────────
    WH  = W * H;
    err = norm(V - WH, 'fro');
    if abs(prev_err - err) / (prev_err + EPS) < tol
        fprintf('  nmf_fp converged at iter %d  err=%.4e\n', iter, err);
        break
    end
    prev_err = err;
end
end
