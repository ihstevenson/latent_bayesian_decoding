
%% Simulate and decode from a single population of tuned neurons receiving latent input

N=20; % population size
trials = 500;

xToZ = @(x) [ones(size(x)) cos(x) sin(x)];  % tuning representation
xGrid = linspace(0,2*pi-2*pi/500, 500)';    % external variable space

trial_x_full = rand(trials,1)*2*pi;     % sample a circular external variable

p = linspace(0,2*pi-2*pi/N,N)';         % evenly spaced preferred directions
b_true = [rand(N,1) randn(N,1).*cos(p) randn(N,1).*sin(p)]; % tuning parameters
lamGrid = exp(xToZ(xGrid)*b_true');     % tuning curves
X = xToZ(trial_x_full);                 % observed covariates

c_true = randn(size(X,1),1);            % sample 1d latent state
d_true = randn(N,1);                    % neuron coefficients for latent state

% simulated firing rates
lambda = exp(zscore(X*b_true') + zscore(c_true*d_true'));

% sample spikes given underlying rate on each trial
trial_y_full = poissrnd(lambda);
    
% get cross-validated decoding results (all models)
res = decode_cv(trial_x_full,trial_y_full,xGrid,xToZ,2,1);

% evaluate coverage
coverage(1) = eval_coverage(res.ols_res_all.Poippp,xGrid,trial_x_full);
coverage(2) = eval_coverage(res.ols_res_all.NBppp,xGrid,trial_x_full);
coverage(3) = eval_coverage(res.lat_res_all.Poippp,xGrid,trial_x_full);
coverage(4) = eval_coverage(res.lat_res_all.NBppp,xGrid,trial_x_full);

%% Visualize simulated spikes and posteriors

[~,sidx] = sort(trial_x_full);

figure(1)
subplot(3,1,1)
imagesc(trial_y_full(sidx,:)')
xlabel('Trials (sorted)')
ylabel('Neurons')
title('Spikes')

subplot(3,1,2)
imagesc(1:trials,xGrid*180/pi,res.ols_res_all.Poippp(:,sidx))
xlabel('Trials (sorted)')
ylabel('Stimulus [deg]')
title('PGLM Posteriors')
hold on
plot(trial_x_full(sidx)*180/pi,'ro')
hold off

subplot(3,1,3)
imagesc(1:trials,xGrid*180/pi,res.lat_res_all.Poippp(:,sidx))
xlabel('Trials (sorted)')
ylabel('Stimulus [deg]')
title('PGLLVM Posteriors')
hold on
plot(trial_x_full(sidx)*180/pi,'ro')
hold off

%% Show coverage

figure(2)
clf
for i=1:4
    plot(coverage(i).credMass_vec,coverage(i).pCover)
    hold on
end
hold off    
xlim([0 1])
ylim([0 1])
line(xlim(),xlim())
legend({'PGLM','NBGLM','PGLLVM','NBGLLVM'})
xlabel('Credible Set')
ylabel('Coverage')

%% Errors and entropy

figure(3)
subplot(2,1,1)
histogram(abs(res.ols_res_all.Poierr)*180/pi,'EdgeColor','none')
hold on
histogram(abs(res.lat_res_all.Poierr)*180/pi,'EdgeColor','none')
hold off
legend({'PGLM','PGLLVM'})
xlabel('Error [deg]')

subplot(2,1,2)
p = res.ols_res_all.Poippp;
h1 = -sum(p.*log2(p));
p = res.lat_res_all.Poippp;
h2 = -sum(p.*log2(p));
histogram(h1,'EdgeColor','none')
hold on
histogram(h2,'EdgeColor','none')
hold off
legend({'PGLM','PGLLVM'})
xlabel('Entropy')