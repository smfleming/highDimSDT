function [post_W] = highdim_SDT_evaluate(X, mu, Sigma, Wprior)
%% Inference on N-D SDT model
%
% SF 2019

%% Initialise variables and conditional prob tables
p_W = Wprior; % prior on perceptual states W

% First compute likelihood of observed X for each possible W (P(X|mu_w, Sigma))
for m = 1:size(mu,1)
    log_lik_X_W(m) = log(mvnpdf(X, mu(m,:), Sigma));
end
log_p_X_W = log_lik_X_W - logsumexp(log_lik_X_W,2); % renormalise to get P(X|W); this step is not really necessary as can work with unnormalised liks

%% Posterior over W (P(W|X=x)
log_post_W = log_p_X_W + log(p_W);  % multiply by prior
log_post_W = log_post_W - logsumexp(log_post_W,2); % normalise

post_W = exp(log_post_W);