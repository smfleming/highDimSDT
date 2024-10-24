clear all
close all

%% Show confidence surfaces for different dimensionalities
% Number of dimensions to evaluate
all_N = [3 5 10 20];
figure;
set(gcf, 'Position', [100 100 1000 300]);

% 3D grid of X's for sensory input on channels 1-3
xgrid = 0:0.05:1;

% Decision rule for confidence - one of 'max', 'difference' or 'normalised'
% - max is Bayesian probability of S1/S2 choice being correct given stimulus
% space (the model that the subject is using)
% - normalised restricts computation to (normalises by) S1/S2 dimensions, ignoring
% irrelevant dimensions (the model the experimenter is using)
% - difference implements heuristic substraction of SX > SY posterior
% probabilities
confidence_rule = 'max';

for dim = 1:length(all_N)

    % Specify dimensionality
    % n - dimensionality of feature space / X
    % m - dimensionality of stimulus space (number of non-zero means, mu)
    % Here we set m=n, which ensures each dimension has one mean associated with
    % it, i.e. each stimulus is unique
    n = all_N(dim);
    m = n;

    % Specify parameters
    mu = eye(m);
    Wprior = repmat(1./m, 1, m);
    Sigma = eye(n);

    % Loop over X1 and X2 holding X3...n constant at 0
    c = 0;    % fixed evidence values on other channels
    for i = 1:length(xgrid)
        for j = 1:length(xgrid)
            for k = 1:length(xgrid)

                X = [xgrid(i) xgrid(j) xgrid(k) ones(1,n-3).*c];
                post_w = highdim_SDT_evaluate(X, mu, Sigma, Wprior);

                % Decision rule for choice
                [y, choice(i,j,k)] = max([post_w(1) post_w(2) post_w(3)]);

                % Decision rule for confidence
                if strcmp(confidence_rule, 'normalised')

                    confW(i,j,k) = max([post_w(1) post_w(2)])./sum([post_w(1) post_w(2)]);  % normalised posterior prob

                elseif strcmp(confidence_rule, 'max')

                    confW(i,j,k) = max([post_w(1) post_w(2) post_w(3)]); % max

                end

            end
        end
    end

    subplot(1,length(all_N),dim)
    slice(xgrid, xgrid, xgrid, confW, [0 1], [1], [0]); % take slices through 3D confidence volume 
    
    box off
    axis square
    colorbar
    xlabel('X1')
    ylabel('X2')
    zlabel('X3')
    title(['k = ' num2str(all_N(dim))])
    set(gca, 'FontSize', 14)

end