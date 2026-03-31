function model = gmm_fp(X, n_components, cov_type, max_iter, tol, reg, random_state)
%GMM_FP  Gaussian Mixture Model via EM.  Mirrors audio_filter/gmm.py.
%
% X           : (n_samples x n_features)
% n_components: number of mixture components k
% cov_type    : 'diag' (default) or 'full'
%
% Returns model struct:
%   .weights  (k x 1)
%   .means    (k x d)
%   .covs     (k x d) if diag  |  (k x d x d) if full
%   .cov_type string
%   .log_lik  scalar — final avg per-sample log-likelihood
%   .n_iter   int

if nargin < 3 || isempty(cov_type),      cov_type      = 'diag'; end
if nargin < 4 || isempty(max_iter),      max_iter      = 100;    end
if nargin < 5 || isempty(tol),           tol           = 1e-4;   end
if nargin < 6 || isempty(reg),           reg           = 1e-6;   end
if nargin < 7 || isempty(random_state),  random_state  = 0;      end

X = double(X);
[n, d] = size(X);
k      = n_components;
EPS    = 1e-10;

% ── k-means++ init ───────────────────────────────────────────────────────────
rng(random_state);
idx     = randi(n);
centers = X(idx, :);
for c = 2:k
    dists = arrayfun(@(i) min(sum((X(i,:) - centers) .^ 2, 2)), 1:n)';
    probs = dists / sum(dists);
    cdf   = cumsum(probs);
    r     = rand();
    idx   = find(cdf >= r, 1);
    centers(end+1, :) = X(idx, :); %#ok<AGROW>
end
means = centers;   % (k x d)

% ── initial covariances ───────────────────────────────────────────────────────
emp_var = var(X, 1, 1) + reg;   % (1 x d)
if strcmp(cov_type, 'diag')
    covs = repmat(emp_var, k, 1);         % (k x d)
else
    if d > 1
        emp_cov = cov(X) * (n-1)/n + reg * eye(d);
    else
        emp_cov = emp_var;
    end
    covs = repmat(reshape(emp_cov, [1 d d]), [k 1 1]);  % (k x d x d)
end

weights = ones(k, 1) / k;
prev_lik = -inf;
avg_lik  = -inf;

for iteration = 1:max_iter
    % ── E-step ───────────────────────────────────────────────────────────────
    lp = zeros(n, k);
    for j = 1:k
        lw = log(max(weights(j), EPS));
        if strcmp(cov_type, 'diag')
            lp(:, j) = lw + log_gauss_diag(X, means(j,:), covs(j,:));
        else
            lp(:, j) = lw + log_gauss_full(X, means(j,:), squeeze(covs(j,:,:)));
        end
    end
    log_liks = logsumexp_rows(lp);            % (n x 1)
    avg_lik  = mean(log_liks);
    log_resp = lp - log_liks;                 % (n x k)

    if abs(avg_lik - prev_lik) < tol, break; end
    prev_lik = avg_lik;

    % ── M-step ───────────────────────────────────────────────────────────────
    resp = exp(log_resp);                     % (n x k)
    Nk   = max(sum(resp, 1)', EPS);           % (k x 1)
    weights = Nk / n;
    means   = (resp' * X) ./ Nk;             % (k x d)

    if strcmp(cov_type, 'diag')
        for j = 1:k
            diff       = X - means(j,:);      % (n x d)
            covs(j,:)  = (resp(:,j)' * (diff.^2)) / Nk(j) + reg;
        end
    else
        for j = 1:k
            diff = X - means(j,:);            % (n x d)
            C    = (repmat(resp(:,j), 1, d) .* diff)' * diff / Nk(j);
            covs(j,:,:) = C + reg * eye(d);
        end
    end
end

model.weights  = weights;
model.means    = means;
model.covs     = covs;
model.cov_type = cov_type;
model.log_lik  = avg_lik;
model.n_iter   = iteration;
end


% ── local helpers ─────────────────────────────────────────────────────────────

function lp = log_gauss_diag(X, mean, var_)
% (n x 1) log N(x ; mean, diag(var_))
d    = size(X, 2);
diff = X - mean;
lp   = -0.5 * (d * log(2*pi) + sum(log(max(var_, 1e-10))) ...
               + sum(diff.^2 ./ max(var_, 1e-10), 2));
end


function lp = log_gauss_full(X, mean, C)
% (n x 1) log N(x ; mean, C) via Cholesky
d    = size(X, 2);
diff = (X - mean)';                          % (d x n)
L    = chol(C, 'lower');
log_det = 2 * sum(log(diag(L)));
y    = L \ diff;                             % (d x n)
mah  = sum(y.^2, 1)';                        % (n x 1)
lp   = -0.5 * (d * log(2*pi) + log_det + mah);
end


function s = logsumexp_rows(A)
% Numerically stable log-sum-exp over columns, returns (n x 1).
a_max = max(A, [], 2);
s     = log(sum(exp(A - a_max), 2)) + a_max;
end
