% Simulate features of confidence for increasing dimensionality
%
%
%
% stephen.fleming@ucl.ac.uk

clear all
close all

addpath('~/Dropbox/Utils/HMeta-d/Matlab/')

%% 1) Show confidence surfaces for different dimensionalities
% Number of dimensions to evaluate
all_N = [2 5 10 20];
figure;
set(gcf, 'Position', [100 100 1000 300]);

% 2D grid of X's for sensory input on channels 1 and 2
xgrid = 0:0.01:1;

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
    c = 0;    % fixed evidence values on other channels
    for i = 1:length(xgrid)
        for j = 1:length(xgrid)

            X = [xgrid(i) xgrid(j) ones(1,n-2).*c];
            post_w = highdim_SDT_evaluate(X, mu, Sigma, Wprior);

            % Decision rule for choice
            [y, choice(i,j)] = max([post_w(1) post_w(2)]);

            % Decision rule for confidence
            if strcmp(confidence_rule, 'normalised')

                confW(i,j,k) = max([post_w(1) post_w(2)])./sum([post_w(1) post_w(2)]);  % normalised posterior prob

            elseif strcmp(confidence_rule, 'max')

                confW(i,j,k) = max([post_w(1) post_w(2)]); % max

            elseif strcmp(confidence_rule, 'difference')

                if choice(i,j) == 1
                    confW(i,j,k) = post_w(1) - post_w(2);
                else
                    confW(i,j,k) = post_w(2) - post_w(1);
                end

            end

        end
    end

    subplot(1,length(all_N),k)
    [C{k},h] = contourf(xgrid, xgrid, confW(:,:,k));
    box off
    axis square
    colorbar
    xlabel('X1')
    ylabel('X2')
    title(['k = ' num2str(all_N(k))])
    set(gca, 'FontSize', 14)

end

%% 2) Quantify angle of the confidence contours with respect to the x and y planes for each dimensionality

for k = 1:length(all_N)

    contourTable = getContourLineCoordinates(C{k});

    for i = 1:max(contourTable.Group)
        xsegment = contourTable.X(contourTable.Group == i);
        ysegment = contourTable.Y(contourTable.Group == i);

        % Take only points above the major diagonal
        above = ysegment >= xsegment;

        dx = gradient(xsegment(above));
        dy = gradient(ysegment(above));
        angles = atan2d(dy, dx); % Angles in degrees

        mean_angle{k}(i) = nanmean(angles); % note that for 2-dim plot, "above" index skips parallel contours 

        if mean_angle{k}(i) < 0
            mean_angle{k}(i) = 180+mean_angle{k}(i);
        end

    end

    all_angles(k) = nanmean(mean_angle{k});
end
disp(all_angles)

%% 3) Why does it happen?
% Take slice through surface to visualise
% Fix X1, vary only X2, evaluate effect of 3rd decoy stimulus

clear choice confW post_w
n = 10;
m = n;

% Specify parameters
xgrid = 0:0.01:1.5;
mu = eye(m);
Wprior = repmat(1./m, 1, m);
Sigma = eye(n);
x2 = 0.75;  % fix evidence value for second channel
eps = 0.01; % add a bit of noise to channels 3:N to allow the lines to be differentiated 
x_means = 0.3+(normrnd(0,eps,1,n-2));   % mean for channels 3:N

for i = 1:length(xgrid)

    X = [xgrid(i) x2 x_means];
    post_w(i,:) = highdim_SDT_evaluate(X, mu, Sigma, Wprior);

    % Decision rule for choice
    [y, choice(i,j)] = max([post_w(i,1) post_w(i,2)]);

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
plot(xgrid, post_w(:,3:end), 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);

box off
set(gca, 'FontSize', 14, 'LineWidth', 1.5)
xlabel('X1')
ylabel('Posterior probability')
legend('p(S1|X)', 'p(S2|X)', 'p(SN|X)')

subplot(1,2,2)
plot(xgrid, confW, 'LineWidth', 2);
box off
set(gca, 'FontSize', 14, 'LineWidth', 1.5)
xlabel('X1')
ylabel('Confidence')

%% 4) Simulate positive evidence effect (regression of X on confidence) as a function of dimensionality
% Draw stimuli from either S1 or S2 Gaussians, store Xchosen and Xunchosen

clear choice confW post_w choice_beta conf_beta all_data

% Maximum number of dimensions to evaluate (2:maxN)
all_N = [2 5 10 20];
figure;
set(gcf, 'Position', [100 100 500 300]);

% Decision rule for confidence - one of 'max', 'difference' or 'normalised'
confidence_rule = 'max';
Ntrials = 10000;

for k = 1:length(all_N)

    % Specify dimensionality
    % n - dimensionality of feature space / X
    % m - dimensionality of stimulus space (number of non-absent means, mu)
    % by setting m=n, can ensure each dimension has one mean associated with
    % it, i.e. each stimulus is unique
    n = all_N(k);
    m = n;

    % Specify parameters
    mu = eye(m);
    Wprior = repmat(1./m, 1, m);
    Sigma = eye(n);

    % Loop over X1 and X2 holding X3...n constant at 0

    for i = 1:Ntrials

        % choose stimulus S1 or S2
        s(i) = (rand < 0.5) + 1;

        % draw sensory samples along n dimensions
        X = mvnrnd(mu(s(i),:), Sigma);

        % invert model
        post_w = highdim_SDT_evaluate(X, mu, Sigma, Wprior);

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

        % store Xchosen, Xunchosen, choice and confidence in a big matrix
        if choice(i) == 1
            Xchosen = X(1);
            Xunchosen = X(2);
            prob_chosen = post_w(1);
            prob_unchosen = post_w(2);
        else
            Xchosen = X(2);
            Xunchosen = X(1);
            prob_chosen = post_w(2);
            prob_unchosen = post_w(1);
        end
        acc(i) = choice(i) == s(i);
        alldata(i,:) = [X(1) X(2) choice(i) Xchosen Xunchosen acc(i) confW(i)];
        allprob(i,:) = [prob_chosen prob_unchosen post_w(3:end)];
    end

    % Fit betas to accuracy and confidence
    choice_beta(:,k) = glmfit(alldata(:,4:5), alldata(:,6), 'binomial', 'link', 'logit');
    conf_beta(:,k) = glmfit(alldata(:,4:5), alldata(:,7));
    mean_probs{k} = mean(allprob);
    clear allprob
end

subplot(1,2,1)
bar(choice_beta(2:3,:), 'k')
box off
set(gca, 'FontSize', 14, 'LineWidth', 1.5)
xticklabels({'X_{chosen}', 'X_{unchosen}'})
ylabel('Beta for accuracy')

subplot(1,2,2)
bar(conf_beta(2:3,:), 'k')
box off
set(gca, 'FontSize', 14, 'LineWidth', 1.5)
xticklabels({'X_{chosen}', 'X_{unchosen}'})
ylabel('Beta for confidence')
