
%% Simulate and decode from a population of tuned neurons receiving latent input

N=20; % population size
r=20;  % number of times to repeat simulation
trials = 500;

xToZ = @(x) [ones(size(x)) cos(x) sin(x)];  % tuning representation
xGrid = linspace(0,2*pi-2*pi/500, 500)';    % external variable space

% simulate multiple times
full_res = cell(0);
for c=1:r
    trial_x_full = rand(trials,1)*2*pi;     % sample a circular external variable
    
    p = linspace(0,2*pi-2*pi/N,N)';         % evenly spaced preferred directions
    b_true = [ones(N,1) ones(N,1).*cos(p) ones(N,1).*sin(p)]; % tuning parameters
    lamGrid = exp(xToZ(xGrid)*b_true');     % tuning curves
    X = xToZ(trial_x_full);                 % observed covariates

    c_true = randn(size(X,1),1);            % sample 1d latent state
    d_true = randn(N,1);                    % neuron coefficients for latent state

    for j=1:4
        if j==1
            % 0% latent variance
            lambda = exp(sqrt(4)*zscore(X*b_true') + 0*zscore(c_true*d_true') - 1);
        elseif j==2
            % 25% latent variance
            lambda = exp(sqrt(12/4)*zscore(X*b_true') + sqrt(4/4)*zscore(c_true*d_true') - 1);
        elseif j==3
            % 50% latent variance
            lambda = exp(sqrt(2)*zscore(X*b_true') + sqrt(2)*zscore(c_true*d_true') - 1);
        elseif j==4
            % 75% latent variance
            lambda = exp(sqrt(4/4)*zscore(X*b_true') + sqrt(12/4)*zscore(c_true*d_true') - 1);
        end

        % sample spikes given underlying rate on each trial
        trial_y_full = poissrnd(lambda);
    
        % get cross-validated decoding results (all models)
        full_res{c,j} = decode_cv(trial_x_full,trial_y_full,xGrid,xToZ,2,1);

        % evaluate coverage
        full_coverage{c,j}(1) = eval_coverage(full_res{c,j}.ols_res_all.Poippp,xGrid,trial_x_full);
        full_coverage{c,j}(2) = eval_coverage(full_res{c,j}.ols_res_all.NBppp,xGrid,trial_x_full);
        full_coverage{c,j}(3) = eval_coverage(full_res{c,j}.lat_res_all.Poippp,xGrid,trial_x_full);
        full_coverage{c,j}(4) = eval_coverage(full_res{c,j}.lat_res_all.NBppp,xGrid,trial_x_full);
    end
end

%% Generate plots for Fig 1C

% collect coverage in a matrix
call=[];
for c=1:r
    for j=1:size(full_coverage,2)
        call(:,:,c,j)= cell2mat(arrayfun(@(x)x.pCover,full_coverage{c,j}','UniformOutput',false))';
    end
end

% plot coverage (averaging over simulations)
variance_labels = {'0%','25%','50%','100%'};
figure(1)
for j=1:size(full_coverage,2)
    subplot(4,1,j)
    plot(full_coverage{1,1}(1).credMass_vec,mean(call(:,:,:,j),3))
    title([variance_labels{j} ' Latent Variance'])
    line(xlim(),xlim())
    box off; set(gca,'TickDir','out')
    xlabel('Credible Set')
    ylabel('Coverage')
end
legend({'PGLM','NBGLM','PGLLVM','NBGLLVM'})

% show multiple variance levels on same plot
figure(2)
clf
subplot(2,1,1)
plot(full_coverage{1,1}(1).credMass_vec,squeeze(mean(call(:,1,:,:),3)))
line(xlim(),xlim())
box off; set(gca,'TickDir','out')
axis equal
xlim([0 1])
ylim([0 1])
xlabel('Credible Set')
ylabel('Coverage')
title('PGLM')
subplot(2,1,2)
plot(full_coverage{1,1}(1).credMass_vec,squeeze(mean(call(:,3,:,:),3)))
line(xlim(),xlim())
box off; set(gca,'TickDir','out')
axis equal
xlim([0 1])
ylim([0 1])
xlabel('Credible Set')
ylabel('Coverage')
title('PGLLVM')
legend(variance_labels)

%% Generate Plots for Fig 1D

% collect error results in matrix
errall=[];
for c=1:r
    for j=1:size(full_coverage,2)
        errall(1,c,j) = median(abs(full_res{c,j}.ols_res_all.Poierr));
        errall(2,c,j) = median(abs(full_res{c,j}.ols_res_all.NBerr));
        errall(3,c,j) = median(abs(full_res{c,j}.lat_res_all.Poierr));
        errall(4,c,j) = median(abs(full_res{c,j}.lat_res_all.NBerr));
    end
end

figure(3)
subplot(2,1,1)
plot(xGrid*180/pi,lamGrid)
xlim([0 360])
box off; set(gca,'TickDir','out')
xlabel('Stimulus Direction [deg]')
ylabel('Firing Rate')

subplot(2,1,2)
bar(squeeze(mean(errall,2))'*180/pi)
hold on
errorbar(squeeze(mean(errall,2))'*180/pi,squeeze(std(errall,1,2))'*180/pi,'o')
hold off
box off; set(gca,'TickDir','out')
xlabel('Latent Variance')
set(gca,'XTickLabel',variance_labels)
ylabel('Median Error [deg]')