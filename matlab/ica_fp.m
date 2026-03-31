function [S, W_full] = ica_fp(X, n_components, max_iter, tol, random_state)
%ICA_FP FastICA source separation.
% Mirrors `audio_filter/ica.py`.
%
% X : (n_channels x n_samples)
%
% Returns:
%   S      (n_components x n_samples)  separated sources
%   W_full (n_components x n_channels) full unmixing matrix

if nargin < 2 || isempty(n_components), n_components = size(X,1);  end
if nargin < 3 || isempty(max_iter),     max_iter     = 500;         end
if nargin < 4 || isempty(tol),          tol          = 1e-5;        end
if nargin < 5 || isempty(random_state), random_state = 0;           end

X = double(X);

% ── center ───────────────────────────────────────────────────────────────────
mu  = mean(X, 2);
Xc  = X - mu;

% ── whiten ───────────────────────────────────────────────────────────────────
[Z, W_w] = local_whiten(Xc);

% ── FastICA ──────────────────────────────────────────────────────────────────
rng(random_state);
W0 = randn(n_components, size(Z,1));
W  = local_sym_orth(W0);

for iter = 1:max_iter
    Y  = W * Z;                         % (n_comp x n_samples)
    G  = tanh(Y);                       % g
    Gp = 1 - G.^2;                      % g'

    W_new = (G * Z') / size(Z,2) - mean(Gp, 2) .* W;
    W_new = local_sym_orth(W_new);

    lim = max(abs(abs(diag(W_new * W')) - 1));
    W   = W_new;
    if lim < tol, break; end
end

W_full = W * W_w;
S      = W_full * Xc;
end


function [Z, W_w] = local_whiten(X)
n = size(X, 2);
C = (X * X') / n;
[V, D] = eig(C);                        % ascending eigenvalues
eigenvalues  = max(diag(D), 1e-10);
[eigenvalues, idx] = sort(eigenvalues, 'descend');
V = V(:, idx);
W_w = diag(eigenvalues .^ -0.5) * V';
Z   = W_w * X;
end


function W = local_sym_orth(W)
[U, ~, V] = svd(W, 'econ');
W = U * V';
end
