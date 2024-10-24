%% Simulate Koizumi
% Add a constant to both stimuli, as in Koizumi et al.
%
%
%
% stephen.fleming@ucl.ac.uk

clear all
close all

% Decision rule for confidence - one of 'max', 'difference' or 'normalised'
confidence_rule = 'max';
Ntrials = 1000;
Nsubjects = 30;

% Specify dimensionality
% n - dimensionality of feature space / X
% m - dimensionality of stimulus space (number of non-absent means, mu)
% by setting m=n, can ensure each dimension has one mean associated with
% it, i.e. each stimulus is unique
n = 10;
m = n;

for diff = 1:2
    
    % Specify parameters
    mu = eye(m);
    Wprior = repmat(1./m, 1, m);
    PEstrength = zeros(m);
    PEstrength(1,1) = 0.5;
    PEstrength(2,2) = 0.5;
    PEstrength(1,2) = 0.5;
    PEstrength(2,1) = 0.5;
    
    % Alter stimulus discriminability
    if diff == 1
        Sigma = eye(n);
    else
        Sigma = eye(n).*2;
    end
    
    % Specify parameters of MoG approximation for inference (see methods of
    % Fleming 2018 for equations)
    muLow = mu;
    muHigh = mu + PEstrength;
    muT = (muLow + muHigh)./2;
    SigmaT = (0.5.*(muLow-muT).^2  + 0.5.*(muHigh-muT).^2)' + Sigma.^2;
    
    % Loop over X1 and X2 holding X3...n constant at 0
    
    for sub = 1:Nsubjects
        
        for i = 1:Ntrials
            
            % choose stimulus S1 or S2
            s(i) = (rand < 0.5) + 1;
            
            % choose PE condition (low or high)
            pe(i) = (rand < 0.5) + 1;
            
            % draw sensory samples along n dimensions
            if pe(i) == 1
                X = mvnrnd(muLow(s(i),:), Sigma);
            else
                X = mvnrnd(muHigh(s(i),:), Sigma);
            end
            
            % invert model
            post_w = highdim_SDT_evaluate(X, muT, SigmaT, Wprior);
            
            % Decision rule for choice
            [y, choice(i)] = max([post_w(1) post_w(2)]);
            
            % Decision rule for confidence
            if strcmp(confidence_rule, 'normalised')
                
                confW(i) = max([post_w(1) post_w(2)])./sum([post_w(1) post_w(2)]);  % normalised posterior prob
                
            elseif strcmp(confidence_rule, 'max')
                
                confW(i) = max([post_w(1) post_w(2)]); % max
                
            elseif strcmp(confidence_rule, 'difference')
                
                if choice(i) == 1
                    confW(i) = post_w(1) - post_w(2);
                else
                    confW(i) = post_w(2) - post_w(1);
                end
                
            end
            
            % store pe condition, accuracy and confidence in big matrix
            acc(i) = choice(i) == s(i);
            
        end
        meanAcc(1,diff,sub) = mean(acc(pe==1));
        meanAcc(2,diff,sub) = mean(acc(pe==2));
        meanConf(1,diff,sub) = mean(confW(pe==1));
        meanConf(2,diff,sub) = mean(confW(pe==2));
        
    end
end

% Get group means and SEs over simulated subjects
meanMeanAcc = mean(meanAcc,3);
meanMeanConf = mean(meanConf,3);
seAcc = std(meanAcc,0,3)./sqrt(Nsubjects);
seConf = std(meanConf,0,3)./sqrt(Nsubjects);

% Plot
figure;
plot(meanMeanAcc(1,:), meanMeanConf(1,:), 'k', 'LineWidth', 3)
hold on
plot(meanMeanAcc(2,:), meanMeanConf(2,:), 'k--', 'LineWidth', 3)
box off
xlabel('Accuracy')
ylabel('Confidence')
legend('Low PE', 'High PE')
set(gca, 'FontSize', 14)
