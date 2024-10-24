%% Maniscalco, Peters & Lau 2016 dissociation simulations as a function of dimensionality
% Present s1 at fixed contrast, but vary s2
% Record d', meta-d' and RC meta-d'
% Plot d' against meta-d'
%
%
%
% stephen.fleming@ucl.ac.uk

clear choice confW post_w alldata

% Maximum number of dimensions to evaluate (2:maxN)
confidence_rule = 'max';
conf_noise = 0.2;

Nsamples = 10000;
mu_S2 = [0.5 0.75 1 1.25 1.5];

% Specify dimensionality
n = 10;
m = n;

% Specify parameters
mu = eye(m);
Wprior = repmat(1./m, 1, m);
Sigma = eye(n);

for i = 1:length(mu_S2)
    
    for n = 1:Nsamples
        
        % choose stimulus S1 or S2
        s(i,n) = (rand < 0.5) + 1;
        
        % draw sensory samples along n dimensions
        if s(i,n) == 1
            mod_mu = mu;
            X = mvnrnd(mod_mu(1,:), Sigma);
        else
            mod_mu = mu;
            mod_mu(2,2) = mu_S2(i);
            X = mvnrnd(mod_mu(2,:), Sigma);
        end
        
        % invert model
        post_w = highdim_SDT_evaluate(X, mod_mu, Sigma, Wprior);
        
        % Decision rule for choice
        [y, choice(i,n)] = max([post_w(1) post_w(2)]);
        
        % Decision rule for confidence
        if strcmp(confidence_rule, 'normalised')
            
            confW(i,n) = max([post_w(1) post_w(2)])./sum([post_w(1) post_w(2)]);  % normalised posterior prob restricted to S1 / S2 only
            
        elseif strcmp(confidence_rule, 'max')
            
            confW(i,n) = max([post_w(1) post_w(2)]); % max
            
        elseif strcmp(confidence_rule, 'difference')
            
            if choice(i) == 1
                confW(i,n) = post_w(1) - post_w(2);
            else
                confW(i,n) = post_w(2) - post_w(1);
            end
            
        end
        
        % store pe condition, accuracy and confidence in big matrix
        acc(i,n) = choice(i,n) == s(i,n);
        conf(i,n) = confW(i,n) + randn.*conf_noise;
    end
end

% Analyse d', meta-d', RC meta-d'
mcmc_params.response_conditional = 0; % Do we want to fit response-conditional meta-d'?
mcmc_params.estimate_dprime = 0;
mcmc_params.nchains = 3; % How Many Chains?
mcmc_params.nburnin = 1000; % How Many Burn-in Samples?
mcmc_params.nsamples = 10000;  %How Many Recorded Samples?
mcmc_params.nthin = 1; % How Often is a Sample Recorded?
mcmc_params.doparallel = 0; % Parallel Option
mcmc_params.dic = 1;  % Save DIC

% Get confidence quantiles
conf_bins = quantile(conf(:), [0.25 0.5 0.75]);
conf_bins = [0 conf_bins 1];

for i = 1:length(mu_S2)
    
    rating = ones(1,Nsamples);
    % Quantize confidnece
    for c = 1:length(conf_bins)-1
        rating(conf(i,:) > conf_bins(c) & conf(i,:) <= conf_bins(c+1)) = c;
    end
    
    [nR_S1, nR_S2] = trials2counts(s(i,:)-1, choice(i,:)-1, rating, length(conf_bins)-1, 0);
    
    mcmc_params.response_conditional = 0;
    fit = fit_meta_d_mcmc(nR_S1, nR_S2, mcmc_params);
    
    dprime(i) = fit.d1;
    meta_d(i) = fit.meta_d;
    
    mcmc_params.response_conditional = 1;
    fit_rc = fit_meta_d_mcmc(nR_S1, nR_S2, mcmc_params);
    meta_d_rS1(i) = fit_rc.meta_d_rS1;
    meta_d_rS2(i) = fit_rc.meta_d_rS2;
    
end

figure;
hold on
plot(dprime, meta_d, 'ko-', 'LineWidth', 2)
plot(dprime, meta_d_rS1, 'ro-', 'LineWidth', 2)
plot(dprime, meta_d_rS2, 'bo-', 'LineWidth', 2)
legend('all', 'S1', 'S2')
xlabel('d')
ylabel('meta-d/d')
box off
set(gca, 'FontSize', 14)