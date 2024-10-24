clear all
close all

%% 1) Show confidence surfaces for different dimensionalities
% Number of dimensions to evaluate
all_N = [2 5 10 20];
figure;
set(gcf, 'Position', [100 100 1000 300]);

% 2D grid of X's for sensory input on channels 1 and 2
xgrid = 0:0.05:3;

% Decision rule for confidence - one of 'max', 'difference' or 'normalised'
% - max is Bayesian probability of S1/S2 choice being correct given stimulus
% space (the model that the subject is using)
% - normalised restricts computation to (normalises by) S1/S2 dimensions, ignoring
% irrelevant dimensions (the model the experimenter is using)
% - difference implements heuristic substraction of SX > SY posterior
% probabilities
confidence_rule = 'max';

for k = 1:length(all_N)
    
    % Specify dimensionality
    % n - dimensionality of feature space / X
    % m - dimensionality of stimulus space (number of non-zero means, mu)
    % Here we set m=n, which ensures each dimension has one mean associated with
    % it, i.e. each stimulus is unique
    n = all_N(k);
    m = n;
    
    % Specify parameters
    mu = eye(m);
    Wprior = repmat(1./m, 1, m);
    Sigma = eye(n);
    
    % Loop over X1 and X2 holding X3...n constant at 0
    c = 1;    % fixed evidence values on other channels
    for i = 1:length(xgrid)
        for j = 1:length(xgrid)
            
            % c = rand(1,n-2);  % could also have randomly varying evidence
            % on other channels to be more realistic 
            X = [xgrid(i) xgrid(j) ones(1,n-2).*c];
            post_w = highdim_SDT_evaluate(X, mu, Sigma, Wprior);
            
            % Decision rule for choice
            [y, choice(i,j)] = max([post_w(1) post_w(2)]);
            
            % Decision rule for confidence
            if strcmp(confidence_rule, 'normalised')
                
                confW(i,j) = max([post_w(1) post_w(2)])./sum([post_w(1) post_w(2)]);  % normalised posterior prob
                
            elseif strcmp(confidence_rule, 'max')
                
                confW(i,j) = max([post_w(1) post_w(2)]); % max
                
            elseif strcmp(confidence_rule, 'difference')
                
                if choice(i,j) == 1
                    confW(i,j) = post_w(1) - post_w(2);
                else
                    confW(i,j) = post_w(2) - post_w(1);
                end
                
            end
            
        end
    end
    
    subplot(1,length(all_N),k)
    contourf(xgrid, xgrid, confW);
    box off
    axis square
    colorbar
    xlabel('X1')
    ylabel('X2')
    title(['k = ' num2str(all_N(k))])
    set(gca, 'FontSize', 14)
    
end

%% 3) Why does it happen?
% Take slice through surface to visualise
% Fix X1, vary only X2, evaluate effect of 3rd decoy stimulus in 3D

clear choice confW post_w
n = 10;
m = n;

% Specify parameters
mu = eye(m);
Wprior = repmat(1./m, 1, m);
Sigma = eye(n);

x2 = 1.5;
x_other = ones(1,n-2) + (rand(1,n-2)./5);    % fixed evidence values on other channels with a little bit of noise to separate out lines 
for i = 1:length(xgrid)
    
    X = [xgrid(i) x2 x_other];
    post_w(i,:) = highdim_SDT_evaluate(X, mu, Sigma, Wprior);
    
    % Decision rule for choice
    [y, choice(i)] = max([post_w(i,1) post_w(i,2)]);
    
    % Decision rule for confidence
    if strcmp(confidence_rule, 'normalised')
        
        confW(i) = max([post_w(i,1) post_w(i,2)])./sum([post_w(i,1) post_w(i,2)]);  % normalised posterior prob
        
    elseif strcmp(confidence_rule, 'max')
        
        confW(i) = max([post_w(i,1) post_w(i,2)]); % max
        
    elseif strcmp(confidence_rule, 'difference')
        
        if choice(i) == 1
            confW(i) = post_w(i,1) - post_w(i,2);
        else
            confW(i) = post_w(i,2) - post_w(i,1);
        end
        
    end
    
end

% figure; plot(xgrid, confW);
figure;
subplot(1,2,1)
plot(xgrid, post_w(:,1), 'LineWidth', 2);
hold on
plot(xgrid, post_w(:,2), 'LineWidth', 2);
plot(xgrid, post_w(:,3:10), 'LineWidth', 1, 'Color', [0.5 0.5 0.5]);
box off
set(gca, 'FontSize', 14, 'LineWidth', 1.5)
xlabel('X1')
ylabel('Posterior probability')
legend('p(S1|X)', 'p(S2|X)', 'p(S3+|X)')

subplot(1,2,2)
plot(xgrid, confW, 'LineWidth', 2);
box off
set(gca, 'FontSize', 14, 'LineWidth', 1.5)
xlabel('X1')
ylabel('Confidence')