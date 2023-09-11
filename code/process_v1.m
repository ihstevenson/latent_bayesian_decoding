
load('../data/data_monkey3_gratings_counts.mat')

xToZ = @(x) [ones(size(x)) cos(x) sin(x) cos(2*x) sin(2*x)];
xGrid = linspace(0,2*pi-2*pi/512, 512)';
z_encode = xToZ(trial_x_full);

% get (not cross-valiated) decoding results from all models
res.ols_res = decode_ols(z_encode,trial_y_full,trial_x_full,trial_y_full,xGrid,xToZ);
res.lat_res = decode_latent(z_encode,trial_y_full,trial_x_full,trial_y_full,1,xGrid,xToZ);
[~,pd] = max(res.ols_res.lam); % preferred directions

% get cross-validated results
res_cv = decode_cv(trial_x_full,trial_y_full,xGrid,xToZ,10,1,1);

%% Fig 5B

offset=10;
cl = [-4 3];

[~,sidx]= sort(trial_x_full*1000+res.lat_res.PoiMod.A);
tmp = trial_y_full(sidx,:);
[~,ridx]=sort(pd,'descend');

figure(22)
clf

subplot(4,1,1)
imagesc(log10(tmp(:,ridx)./exp(res.ols_res.Poi_BETA_ols(1,ridx))+1)')
colormap turbo
ylabel('Neuron [sorted]')
colorbar
box off; set(gca,'TickDir','out')

subplot(4,1,2)
lamg = exp(z_encode(sidx,:)*res.ols_res.Poi_BETA_ols(:,ridx));
imagesc(log(lamg./exp(res.ols_res.Poi_BETA_ols(1,ridx)))')
title('GLM')
colorbar
set(gca,'CLim',cl)
box off; set(gca,'TickDir','out')

subplot(4,1,3)
lam = exp(z_encode(sidx,:)*res.lat_res.PoiMod.BETA(:,ridx) + res.lat_res.PoiMod.A(sidx)*res.lat_res.PoiMod.B(ridx));
imagesc(log(lam./exp(res.ols_res.Poi_BETA_ols(1,ridx)))')
title('GLLVM')
colorbar
set(gca,'CLim',cl)
box off; set(gca,'TickDir','out')

subplot(4,1,4)
stairs(sum(log10(tmp+offset),2))
hold on
stairs(sum(log10(lamg+offset),2))
stairs(sum(log10(lam+offset),2))
hold off
axis tight
box off; set(gca,'TickDir','out')
ylabel('Population Activity')
legend({'Observed','GLM','GLLVM'})
xlabel('Trials [sorted]')
colorbar

%% Get stimulus and noise correlations for data and models

N = size(trial_y_full,2);
Cobs = corrcoef(tmp(:,ridx))-eye(N);
Clamg = zeros(N,N,100);
Clam = zeros(N,N,100);
for i=1:size(Clamg,3)
    Clamg(:,:,i) = corrcoef(poissrnd(lamg))-eye(N);
    Clam(:,:,i) = corrcoef(poissrnd(lam))-eye(N);
end

Cstim = zeros(N,N,100);
Clamg_stim = zeros(N,N,100);
Clam_stim = zeros(N,N,100);
for i=1:size(Clamg,3)
    Cstim(:,:,i) = corrcoef(shuffleTrials(tmp(:,ridx),trial_x_full(sidx)))-eye(N);
    Clamg_stim(:,:,i) = corrcoef(poissrnd(shuffleTrials(lamg,trial_x_full(sidx))))-eye(N);
    Clam_stim(:,:,i) = corrcoef(poissrnd(shuffleTrials(lam,trial_x_full(sidx))))-eye(N);
end


%% Fig 5B - correlations

cl1 = [-.6 .6];
cl2 = [-.4 .4];
cl3 = [-.3 .3];

D = squareform(pdist(xGrid(pd(ridx)),'circ_dist'));

figure(23)
subplot(4,3,1)
imagesc(Cobs)
axis equal; box off; set(gca,'TickDir','out')
colorbar
set(gca,'CLim',cl1)

subplot(4,3,4)
imagesc(mean(Clamg,3))
axis equal; box off; set(gca,'TickDir','out')
colorbar
set(gca,'CLim',cl1)

subplot(4,3,7)
imagesc(mean(Clam,3))
axis equal; box off; set(gca,'TickDir','out')
colorbar
set(gca,'CLim',cl1)

subplot(4,3,2)
imagesc(mean(Cstim,3))
axis equal; box off; set(gca,'TickDir','out')
colorbar
set(gca,'CLim',cl2)

subplot(4,3,5)
imagesc(mean(Clamg_stim,3))
axis equal; box off; set(gca,'TickDir','out')
colorbar
set(gca,'CLim',cl2)

subplot(4,3,8)
imagesc(mean(Clam_stim,3))
axis equal; box off; set(gca,'TickDir','out')
colorbar
set(gca,'CLim',cl2)

subplot(4,3,3)
imagesc(Cobs-mean(Cstim,3))
axis equal; box off; set(gca,'TickDir','out')
colorbar
set(gca,'CLim',cl3)

subplot(4,3,6)
imagesc(mean(Clamg,3)-mean(Clamg_stim,3))
axis equal; box off; set(gca,'TickDir','out')
colorbar
set(gca,'CLim',cl3)

subplot(4,3,9)
imagesc(mean(Clam,3)-mean(Clam_stim,3))
axis equal; box off; set(gca,'TickDir','out')
colorbar
set(gca,'CLim',cl3)

subplot(4,3,10)
plotSegmentErrorBar(abs(D(:)),Cobs(:),32,true)
hold on
cc = mean(Clamg,3);
plotSegmentErrorBar(abs(D(:)),cc(:),32,false)
cc = mean(Clam,3);
plotSegmentErrorBar(abs(D(:)),cc(:),32,false)
hold off
colorbar
box off; set(gca,'TickDir','out')

subplot(4,3,11)
plotSegmentErrorBar(abs(D(~eye(N))),Cstim(~eye(N)),32,true)
hold on
cc = mean(Clamg_stim,3);
plotSegmentErrorBar(abs(D(~eye(N))),cc(~eye(N)),32,false)
cc = mean(Clam_stim,3);
plotSegmentErrorBar(abs(D(~eye(N))),cc(~eye(N)),32,false)
hold off
colorbar
box off; set(gca,'TickDir','out')

subplot(4,3,12)
cc= Cobs-mean(Cstim,3);
plotSegmentErrorBar(abs(D(~eye(N))),cc(~eye(N)),32,true)
hold on
cc = mean(Clamg,3)-mean(Clamg_stim,3);
plotSegmentErrorBar(abs(D(~eye(N))),cc(~eye(N)),32,false)
cc = mean(Clam,3)-mean(Clam_stim,3);
plotSegmentErrorBar(abs(D(~eye(N))),cc(~eye(N)),32,false)
hold off
colorbar
box off; set(gca,'TickDir','out')

%% Fig 3B - example tuning curves

zplot = xToZ(xGrid);
lam = exp(zplot*res.ols_res.Poi_BETA_ols);

figure(24)
ridx = [1 2 3 4 5];
for r=1:length(ridx)
    subplot(1,5,r)
    plot(xGrid*180/pi,exp(zplot*res.ols_res.Poi_BETA_ols(:,ridx(r))))
    set(gca,'TickDir','out')
    box off
    axis tight
end


%% Fig 3E - example and average posteriors

figure(25)
subplot(1,2,1)
ridx = 200:205;
for i=1:length(ridx)
    [~,j] = min(abs(xGrid-trial_x_full(ridx(i))));
    ptmp = circshift(res_cv.ols_res_all.Poippp(:,ridx(i)),256-j);
    plot((xGrid-pi)*180/pi,ptmp,'b')
    hold on

    ptmp = circshift(res_cv.lat_res_all.NBppp(:,ridx(i)),256-j);
    plot((xGrid-pi)*180/pi,ptmp,'r')
end
hold off
set(gca,'TickDir','out')
box off
xlim([-20 20])


palign=[];
for i=1:length(trial_x_full)
    [~,j] = max(res_cv.ols_res_all.Poippp(:,i));
    palign(:,i,1) = circshift(res_cv.ols_res_all.Poippp(:,i),256-j);
    [~,j] = max(res_cv.lat_res_all.NBppp(:,i));
    palign(:,i,2) = circshift(res_cv.lat_res_all.NBppp(:,i),256-j);
end
subplot(1,2,2)
plot((xGrid-pi)*180/pi,mean(palign(:,:,1),2),'b')
hold on
plot((xGrid-pi)*180/pi,mean(palign(:,:,2),2),'r')
hold off
set(gca,'TickDir','out')
box off
xlim([-20 20])


%% Fig 4B

coverVec = linspace(0.001,0.999,256);
coverage(1) = eval_coverage(res_cv.ols_res_all.Poippp,xGrid,trial_x_full,coverVec);
coverage(2) = eval_coverage(res_cv.ols_res_all.NBppp,xGrid,trial_x_full,coverVec);
coverage(3) = eval_coverage(res_cv.lat_res_all.Poippp,xGrid,trial_x_full,coverVec);
coverage(4) = eval_coverage(res_cv.lat_res_all.NBppp,xGrid,trial_x_full,coverVec);

figure(21)
for i=1:4
    plot(coverage(i).credMass_vec,coverage(i).pCover_corr)
    hold on
end
hold off    
xlim([0 1])
ylim([0 1])
line(xlim(),xlim())
legend({'PGLM','NBGLM','PGLLVM','NBGLLVM'})
xlabel('Credible Set')
ylabel('Coverage')

%% Get post-hoc corrected posteriors

coverVec = linspace(0.01,0.99,20);
pall=res_cv.ols_res_all.Poippp;
coverCost = @(c) sum(coverage_r(normlogP((1./(1+exp(-c)))*log(pall')),xGrid,trial_x_full,coverVec)-coverVec).^2;
c = fminsearch(coverCost,-1);
cxpall_pdglm = normlogP((1./(1+exp(-c)))*log(pall'));
call(1)=c;

pall=res_cv.ols_res_all.NBppp;
coverCost = @(c) sum(coverage_r(normlogP((1./(1+exp(-c)))*log(pall')),xGrid,trial_x_full,coverVec)-coverVec).^2;
c = fminsearch(coverCost,-1);
cxpall_nbdglm = normlogP((1./(1+exp(-c)))*log(pall'));
call(2)=c;

pall=res_cv.lat_res_all.Poippp;
coverCost = @(c) sum(coverage_r(normlogP((1./(1+exp(-c)))*log(pall')),xGrid,trial_x_full,coverVec)-coverVec).^2;
c = fminsearch(coverCost,-1);
cxpall_pdfa = normlogP((1./(1+exp(-c)))*log(pall'));
call(3)=c;

pall=res_cv.lat_res_all.NBppp;
coverCost = @(c) sum(coverage_r(normlogP((1./(1+exp(-c)))*log(pall')),xGrid,trial_x_full,coverVec)-coverVec).^2;
c = fminsearch(coverCost,-1);
cxpall_nbdfa = normlogP((1./(1+exp(-c)))*log(pall'));
call(4)=c;


coverVec = linspace(0.001,0.999,256);
cxp_coverage(1) = eval_coverage(cxpall_pdglm,xGrid,trial_x_full,coverVec);
cxp_coverage(2) = eval_coverage(cxpall_nbdglm,xGrid,trial_x_full,coverVec);
cxp_coverage(3) = eval_coverage(cxpall_pdfa,xGrid,trial_x_full,coverVec);
cxp_coverage(4) = eval_coverage(cxpall_nbdfa,xGrid,trial_x_full,coverVec);

% Fig 7B
figure(26)
for i=1:4
    plot(cxp_coverage(i).credMass_vec,cxp_coverage(i).pCover)
    hold on
end
hold off    
xlim([0 1])
ylim([0 1])
line(xlim(),xlim())
legend({'PGLM','NBGLM','PGLLVM','NBGLLVM'})
xlabel('Credible Set')
ylabel('Coverage')

%% Fig 9B

figure(27)
clf

pall = res_cv.ols_res_all.Poippp;
err = res_cv.ols_res_all.Poierr;
err(err==0)=0.1*pi/180;
cstd=[];
for i=1:size(pall,2)
    cstd(i) = circ_std(xGrid,pall(:,i));
end

subplot(2,1,1)
histogram(log10(cstd*180/pi),'EdgeColor','none')
box off; set(gca,'TickDir','out')
subplot(2,1,2)
edges = prctile(log10(cstd*180/pi),linspace(0,100,11));
x0 = edges(1:end-1)+diff(edges)/2;
[~,~,bins] = histcounts(log10(cstd*180/pi),edges);
hhm=[];
for i=1:length(edges)-1
    if sum(bins==i)>0
        hhm(i,1) = mean(log10(abs(err(bins==i))*180/pi));
        hhm(i,2) = std(log10(abs(err(bins==i))*180/pi));
    else
        hhm(i,:)=NaN;
    end
end
errorbar(x0,hhm(:,1),hhm(:,2),'o')
box off; set(gca,'TickDir','out')
hold on
b = glmfit(log10(cstd*180/pi),log10(abs(err)*180/pi),'normal');
xl=xlim();
x00 = linspace(xl(1),xl(2),256);
yhat = glmval(b,x00,'identity');
plot(x00,yhat)

% add NB GLLVM
pall = res_cv.lat_res_all.NBppp;
err = res_cv.lat_res_all.NBerr;
err(err==0)=0.1*pi/180;
cstd=[];
for i=1:size(pall,2)
    cstd(i) = circ_std(xGrid,pall(:,i));
end


subplot(2,1,1)
hold on
histogram(log10(cstd*180/pi),'EdgeColor','none')
subplot(2,1,2)
edges = prctile(log10(cstd*180/pi),linspace(0,100,11));
x0 = edges(1:end-1)+diff(edges)/2;
[~,~,bins] = histcounts(log10(cstd*180/pi),edges);
hhm=[];
for i=1:length(edges)-1
    if sum(bins==i)>0
        hhm(i,1) = mean(log10(abs(err(bins==i))*180/pi));
        hhm(i,2) = std(log10(abs(err(bins==i))*180/pi));
    else
        hhm(i,:)=NaN;
    end
end
hold on
errorbar(x0,hhm(:,1),hhm(:,2),'o')
hold on
b = glmfit(log10(cstd*180/pi),log10(abs(err)*180/pi),'normal');
xl=xlim();
x00 = linspace(xl(1),xl(2),256);
yhat = glmval(b,x00,'identity');
plot(x00,yhat)

% add post-hoc corrected PGLM
pall = cxpall_pdglm;
err = res_cv.ols_res_all.Poierr;
err(err==0)=0.1*pi/180;
cstd=[];
for i=1:size(pall,2)
    cstd(i) = circ_std(xGrid,pall(:,i));
end
subplot(2,1,1)
histogram(log10(cstd*180/pi),'EdgeColor','none')
box off; set(gca,'TickDir','out')
subplot(2,1,2)

edges = prctile(log10(cstd*180/pi),linspace(0,100,11));
x0 = edges(1:end-1)+diff(edges)/2;
[~,~,bins] = histcounts(log10(cstd*180/pi),edges);
hhm=[];
for i=1:length(edges)-1
    if sum(bins==i)>0
        hhm(i,1) = mean(log10(abs(err(bins==i))*180/pi));
        hhm(i,2) = std(log10(abs(err(bins==i))*180/pi));
    else
        hhm(i,:)=NaN;
    end
end
errorbar(x0,hhm(:,1),hhm(:,2),'o')
box off; set(gca,'TickDir','out')
hold on

[b,dev,stat] = glmfit(log10(cstd*180/pi),log10(abs(err)*180/pi),'normal');
xl=xlim();
x00 = linspace(xl(1),xl(2),256);
yhat = glmval(b,x00,'identity');
plot(x00,yhat)