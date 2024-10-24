%% Compute sample variance for evidence and confidence
%
%
% stephen.fleming@ucl.ac.uk

clear all
close all

% Maximum number of dimensions to evaluate (2:maxN)
all_N = [2 5 10 20];

% Decision rule for confidence - one of 'max', 'difference' or 'normalised'
confidence_rule = 'max';
Ntrials = 2500; % trials per run
Nsimulations = 10;
plotsim = 1;    % plot individual runs of the simulation

if plotsim
    figure;
    set(gcf, 'Position', [100 100 1000 200]);
end

for k = 1:length(all_N)
    for sim = 1:Nsimulations
        
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
        Sigma = eye(n).*0.5;
        
        % Loop over X1 and X2 holding X3...n constant at 0
        
        for i = 1:Ntrials
            
            % choose stimulus S1 or S2
            s(i) = (rand < 0.5) + 1;
            
            % draw sensory samples along n dimensions
            X = mvnrnd(mu(s(i),:), Sigma);
            X1(i) = X(1);
            X2(i) = X(2);
            
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
            
        end
        
        if plotsim
            % Plot samples with colour governed by confidence
            subplot(1,length(all_N),k)
            scatter1 = scatter(X1(s==1), X2(s==1), 10, confW(s==1), 'filled');
            hold on
            scatter2 = scatter(X1(s==2), X2(s==2), 10, confW(s==2), 'filled');

            % Set two different colormaps for confidence to separate out the distributions
            colormap1 = autumn;
            cdata1 = interp1(linspace(min(confW(s==1)), max(confW(s==1)), size(colormap1, 1)), colormap1, confW(s==1));
            set(scatter1, 'CData', cdata1);

            % For the second scatter plot, we will use the 'parula' colormap
            colormap2 = winter;
            cdata2 = interp1(linspace(min(confW(s==2)), max(confW(s==2)), size(colormap2, 1)), colormap2, confW(s==2));
            set(scatter2, 'CData', cdata2);

            box off
            axis square
            xlabel('X1')
            ylabel('X2')
            title(['k = ' num2str(all_N(k))])
            set(gca, 'FontSize', 14, 'XLim', [-3 5], 'YLim', [-3 5])
        end
        %% Take slices through plot on target and non-target dimensions to calculate confidence variance (use s2 as target as symmetric in s1)
        range_of_X = [-0.05 0.05];
        points_to_select = X1 >= range_of_X(1) & X1 <= range_of_X(2);
        conf_var_target(k,sim) = std(confW(points_to_select & s == 2));
        x_var_target(k,sim) = std(X2(points_to_select & s == 2));
        
        range_of_X = [0.95 1.05];
        points_to_select = X2 >= range_of_X(1) & X2 <= range_of_X(2);
        conf_var_nontarget(k,sim) = std(confW(points_to_select & s == 2));
        x_var_nontarget(k,sim) = std(X1(points_to_select & s == 2));
    end
end

% figure; plot(all_N, conf_var_target)
% hold on;
% plot(all_N, conf_var_nontarget, 'r')
% box off

conf_var_ratio = conf_var_target./conf_var_nontarget;

figure;
errorbar(all_N, mean(conf_var_ratio'), std(conf_var_ratio')./sqrt(Nsimulations), 'LineWidth', 2);
box off
xlabel('Dimensionality')
ylabel('Target / nontarget variance in confidence')
set(gca, 'FontSize', 14, 'YLim', [0.6 3.5], 'XLim', [0 21])

x_var_ratio = x_var_target./x_var_nontarget;

figure;
errorbar(all_N, mean(x_var_ratio'), std(x_var_ratio')./sqrt(Nsimulations), 'LineWidth', 2);
box off
xlabel('Dimensionality')
ylabel('Target / nontarget variance in X')
set(gca, 'FontSize', 14, 'YLim', [0.6 3.5], 'XLim', [0 21])
