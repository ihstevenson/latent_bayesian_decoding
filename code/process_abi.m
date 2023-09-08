

load abi_742951821_natscenes
trial_y_full=trial_y_full(:,mean(trial_y_full)>1);

xToZref = -1:max(trial_x_full);
xToZ = @(x) [ones(size(x)) double(xToZref==x)];
xGrid = [-1:max(trial_x_full)]';
N = size(trial_y_full,2);

x_encode = trial_x_full;
x_decode = trial_x_full;
y_encode = trial_y_full;
y_decode = trial_y_full;
z_encode = xToZ(x_encode);

res = decode_cv_cat(trial_x_full,trial_y_full,xGrid,xToZ,10,1,100);

%% coverage plot
pCover=[];
coverVec = linspace(0.001,0.999,256);
coverage = eval_coverage_cat(res.ols_res_all.Poippp,xGrid,trial_x_full,coverVec);
pCover(1,:) = coverage.pCover_corr;
coverage = eval_coverage_cat(res.ols_res_all.NBppp,xGrid,trial_x_full,coverVec);
pCover(2,:) = coverage.pCover_corr;
coverage = eval_coverage_cat(res.lat_res_all.Poippp,xGrid,trial_x_full,coverVec);
pCover(3,:) = coverage.pCover_corr;
coverage = eval_coverage_cat(res.lat_res_all.NBppp,xGrid,trial_x_full,coverVec);
pCover(4,:) = coverage.pCover_corr;

%%
figure(23)
plot(coverVec,pCover)
set(gca,'TickDir','out')
box off
xlim([0 1]); ylim([0 1])
line(xlim(),xlim())

%%

offset=10;
% cl = [log10(offset) 1.65];
cl = [-3 2];
xl = [0 817];

% [~,sidx]=sort(trial_x_full);
[~,sidx]= sort(trial_x_full*10000+res.lat_res.PoiMod.A(:,1));
% z = sum(log10(trial_y_full+1),2);
% [~,sidx]= sort(trial_x_full*100000+z,'descend');
tmp = trial_y_full(sidx,:);


% [~,ridx]=sort(mean(trial_y_full),'descend');
[~,ridx]=sort(pd,'descend');
[iclustup, ridx] = embed1D(zscore(trial_y_full)', 32, 1:size(trial_y_full,2), false);

figure(1)
clf

subplot(4,1,1)
% imagesc(log10(tmp(:,ridx)+1))
% imagesc((tmp(:,ridx)./exp(res.ols_res.Poi_BETA_ols(1,ridx))))
imagesc(log10(tmp(:,ridx)./exp(res.ols_res.Poi_BETA_ols(1,ridx))+1)')
colormap turbo
ylabel('Neuron [sorted]')
% xlabel('Trials [sorted]')
% title('Spike Counts')
colorbar
% cl = get(gca,'CLim');
% set(gca,'CLim',cl)
xlim(xl)
box off; set(gca,'TickDir','out')

subplot(4,1,2)
lamg = exp(z_encode(sidx,:)*res.ols_res.Poi_BETA_ols(:,ridx));
imagesc(log(lamg./exp(res.ols_res.Poi_BETA_ols(1,ridx)))')
% title('GLM')
colorbar
set(gca,'CLim',cl)
xlim(xl)
box off; set(gca,'TickDir','out')

subplot(4,1,3)
lam = exp(z_encode(sidx,:)*res.lat_res.PoiMod.BETA(:,ridx) + res.lat_res.PoiMod.A(sidx,:)*res.lat_res.PoiMod.B(:,ridx));
imagesc(log(lam./exp(res.ols_res.Poi_BETA_ols(1,ridx)))')
% title('GLLVM')
colorbar
set(gca,'CLim',cl)
xlim(xl)
box off; set(gca,'TickDir','out')

% figure(2)
% clf
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
xlim(xl)

%% get stimulus and noise correlations for data and models

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


%%

cl1 = [-.6 .6];
cl2 = [-.03 .03];
cl3 = [-.3 .3];

% D = squareform(pdist([xGrid(pd(ridx))],'circ_dist'));

N = size(y_encode,2);
figure(3)
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
% plot(abs(D(:)),Cobs(:),'o')
plotSegmentErrorBar(abs(D(:)),Cobs(:),32,true)
hold on
cc = mean(Clamg,3);
% plot(abs(D(:)),cc(:),'o')
plotSegmentErrorBar(abs(D(:)),cc(:),32,false)
cc = mean(Clam,3);
% plot(abs(D(:)),cc(:),'o')
plotSegmentErrorBar(abs(D(:)),cc(:),32,false)
hold off
colorbar
box off; set(gca,'TickDir','out')
%
% subplot(4,3,11)
% % plot(abs(D(:)),Cobs(:),'o')
% plotSegmentErrorBar(abs(D(~eye(N))),Cstim(~eye(N)),32,true)
% hold on
% cc = mean(Clamg_stim,3);
% % plot(abs(D(:)),cc(:),'o')
% plotSegmentErrorBar(abs(D(~eye(N))),cc(~eye(N)),32,false)
% cc = mean(Clam_stim,3);
% % plot(abs(D(:)),cc(:),'o')
% plotSegmentErrorBar(abs(D(~eye(N))),cc(~eye(N)),32,false)
% hold off
% colorbar
% box off; set(gca,'TickDir','out')
%
% subplot(4,3,12)
% % plot(abs(D(:)),Cobs(:),'o')
% cc= Cobs-mean(Cstim,3);
% plotSegmentErrorBar(abs(D(~eye(N))),cc(~eye(N)),32,true)
% hold on
% cc = mean(Clamg,3)-mean(Clamg_stim,3);
% % plot(abs(D(:)),cc(:),'o')
% plotSegmentErrorBar(abs(D(~eye(N))),cc(~eye(N)),32,false)
% cc = mean(Clam,3)-mean(Clam_stim,3);
% % plot(abs(D(:)),cc(:),'o')
% plotSegmentErrorBar(abs(D(~eye(N))),cc(~eye(N)),32,false)
% hold off
% colorbar
% box off; set(gca,'TickDir','out')

%%


coverVec = linspace(0.01,0.99,20);

pall=res.ols_res_all.Poippp;
% coverCost = @(c) sum(coverage_cat(normlogP((1./(1+exp(-c)))*log(pall')),xGrid,x_encode,coverVec,1)-coverVec).^2;
coverCost = @(c) sum(coverage_cat(normlogP((exp(c))*log(pall')),xGrid,x_encode,coverVec,1)-coverVec).^2;
c = fminsearch(coverCost,0);
% cxpall_pdglm = normlogP((1./(1+exp(-c)))*log(pall'));
cxpall_pdglm = normlogP((exp(c))*log(pall'));
call(1)=c;

pall=res.lat_res_all.Poippp;
% coverCost = @(c) sum(coverage_cat(normlogP((1./(1+exp(-c)))*log(pall')),xGrid,x_encode,coverVec,1)-coverVec).^2;
coverCost = @(c) sum(coverage_cat(normlogP((exp(c))*log(pall')),xGrid,x_encode,coverVec,1)-coverVec).^2;
c = fminsearch(coverCost,0);
% cxpall_pdfa = normlogP((1./(1+exp(-c)))*log(pall'));
cxpall_pdfa = normlogP((exp(c))*log(pall'));
call(2)=c;

pall=res.ols_res_all.NBppp;
% coverCost = @(c) sum(coverage_cat(normlogP((1./(1+exp(-c)))*log(pall')),xGrid,x_encode,coverVec,1)-coverVec).^2;
coverCost = @(c) sum(coverage_cat(normlogP((exp(c))*log(pall')),xGrid,x_encode,coverVec,1)-coverVec).^2;
c = fminsearch(coverCost,0);
% cxpall_nbdglm = normlogP((1./(1+exp(-c)))*log(pall'));
cxpall_nbdglm = normlogP((exp(c))*log(pall'));
call(3)=c;

pall=res.lat_res_all.NBppp;
% coverCost = @(c) sum(coverage_cat(normlogP((1./(1+exp(-c)))*log(pall')),xGrid,x_encode,coverVec,1)-coverVec).^2;
coverCost = @(c) sum(coverage_cat(normlogP((exp(c))*log(pall')),xGrid,x_encode,coverVec,1)-coverVec).^2;
c = fminsearch(coverCost,0);
% cxpall_nbdfa = normlogP((1./(1+exp(-c)))*log(pall'));
cxpall_nbdfa = normlogP((exp(c))*log(pall'));
call(4)=c;


cxpCover=[];
coverVec = linspace(0.001,0.999,256);
coverage = eval_coverage_cat(cxpall_pdglm,xGrid,x_encode,coverVec);
cxpCover(1,:) = coverage.pCover_corr;
coverage = eval_coverage_cat(cxpall_pdfa,xGrid,x_encode,coverVec);
cxpCover(2,:) = coverage.pCover_corr;
coverage = eval_coverage_cat(cxpall_nbdglm,xGrid,x_encode,coverVec);
cxpCover(3,:) = coverage.pCover_corr;
coverage = eval_coverage_cat(cxpall_nbdfa,xGrid,x_encode,coverVec);
cxpCover(4,:) = coverage.pCover_corr;

figure(24)
plot(coverVec,cxpCover)
set(gca,'TickDir','out')
box off
xlim([0 1]); ylim([0 1])
line(xlim(),xlim())

%%

% tuning curve examples
zplot = xToZ(xGrid);
lam = exp(zplot*res.ols_res(1).Poi_BETA_ols);
figure(3)
ridx = [24 15 21 41 239];
for r=1:length(ridx)
    subplot(1,5,r)
    stairs(lam(:,ridx(r)))
    %     hold on
    set(gca,'TickDir','out')
    box off
    axis tight
end
% hold off

%%


figure(2)
clf
ridx = [10 11 14 27];
imsub = unique([x_encode(ridx)+2; [1:14]']);
for i=1:length(ridx)
    subplot(length(ridx),1,i)
    bar(res.ols_res_all.Poippp(imsub,ridx(i)),1,'EdgeColor','flat','FaceColor','b')
    hold on
    bar(res.lat_res_all.NBppp(imsub,ridx(i)),1,'EdgeColor','flat','FaceColor','r')

    plot(find(imsub==x_encode(ridx(i))+2,1),0,'^')
    hold off
    set(gca,'TickDir','out')
    box off
end

%%

palign=[];
for i=1:length(x_decode)
    palign(:,i,1) = sort(res.ols_res_all.Poippp(:,i),'descend');
    palign(:,i,2) = sort(res.lat_res_all.NBppp(:,i),'descend');
end

figure(4)
bar(mean(palign(:,:,1),2),1,'EdgeColor','flat','FaceColor','b')
hold on
bar(mean(palign(:,:,2),2),1,'EdgeColor','flat','FaceColor','r')
hold off
set(gca,'TickDir','out')
box off

%%

[~,mapi] = max(res.ols_res_all.Poippp);
ee = trial_x_full==xGrid(mapi);
[phat,ci]=binofit(sum(ee),length(ee))
hh = nansum(-res.ols_res_all.Poippp.*log2(res.ols_res_all.Poippp))';

edges = linspace(0,log2(length(xGrid)),32);
figure(13)
clf
subplot(2,1,1)
histogram(hh,edges,'EdgeColor','none')
box off; set(gca,'TickDir','out')

subplot(2,1,2)
x0 = edges+mean(diff(edges))/2;
edges = prctile(hh,linspace(0,100,11));
x0 = edges(1:end-1)+diff(edges)/2;
[~,~,bins] = histcounts(hh,edges);
hhm=[];
for i=1:length(edges)-1
    if sum(bins==i)>0
        [hhm(i,1),hhm(i,2:3)] = binofit(sum(ee(bins==i)),length(ee(bins==i)));
    else
        hhm(i,:)=NaN;
    end
end
errorbar(x0,hhm(:,1),hhm(:,1)-hhm(:,2),hhm(:,3)-hhm(:,1),'o')
hold on

[b,dev,stat] = glmfit(hh,ee,'binomial');
x00 = linspace(0,log2(length(xGrid)),256);
[yhat,ylo,yhi] = glmval(b,x00,'logit',stat);
plot(x00,yhat)

hold off
box off; set(gca,'TickDir','out')



[~,mapi] = max(res.lat_res_all.NBppp);
ee = trial_x_full==xGrid(mapi);
hh = nansum(-res.lat_res_all.NBppp.*log2(res.lat_res_all.NBppp))';

edges = linspace(0,log2(length(xGrid)),32);
figure(13)
subplot(2,1,1)
hold on
histogram(hh,edges,'EdgeColor','none')
box off; set(gca,'TickDir','out')
hold off

subplot(2,1,2)
x0 = edges+mean(diff(edges))/2;
edges = prctile(hh,linspace(0,100,11));
x0 = edges(1:end-1)+diff(edges)/2;
[~,~,bins] = histcounts(hh,edges);
hhm=[];
for i=1:length(edges)-1
    if sum(bins==i)>0
        [hhm(i,1),hhm(i,2:3)] = binofit(sum(ee(bins==i)),length(ee(bins==i)));
    else
        hhm(i,:)=NaN;
    end
end
hold on
errorbar(x0,hhm(:,1),hhm(:,1)-hhm(:,2),hhm(:,3)-hhm(:,1),'o')


[b,dev,stat] = glmfit(hh,ee,'binomial');
x00 = linspace(0,log2(length(xGrid)),256);
[yhat,ylo,yhi] = glmval(b,x00,'logit',stat);
plot(x00,yhat)

hold off
box off; set(gca,'TickDir','out')

%% add corrected

[~,mapi] = max(res.ols_res_all.Poippp);
ee = trial_x_full==xGrid(mapi);
% hh = nansum(-res.lat_res_all.NBppp.*log2(res.lat_res_all.NBppp))';
q = getCorrectedP(res.ols_res_all.Poippp,xGrid,x_encode);
hh = nansum(-q.*log2(q))';

edges = linspace(0,log2(length(xGrid)),32);
figure(13)
subplot(2,1,1)
hold on
histogram(hh,edges,'EdgeColor','none')
box off; set(gca,'TickDir','out')
hold off

subplot(2,1,2)
x0 = edges+mean(diff(edges))/2;
edges = prctile(hh,linspace(0,100,11));
x0 = edges(1:end-1)+diff(edges)/2;
[~,~,bins] = histcounts(hh,edges);
hhm=[];
for i=1:length(edges)-1
    if sum(bins==i)>0
        [hhm(i,1),hhm(i,2:3)] = binofit(sum(ee(bins==i)),length(ee(bins==i)));
    else
        hhm(i,:)=NaN;
    end
end
hold on
errorbar(x0,hhm(:,1),hhm(:,1)-hhm(:,2),hhm(:,3)-hhm(:,1),'o')


[b,dev,stat] = glmfit(hh,ee,'binomial');
x00 = linspace(0,log2(length(xGrid)),256);
[yhat,ylo,yhi] = glmval(b,x00,'logit',stat);
plot(x00,yhat)

hold off
box off; set(gca,'TickDir','out')