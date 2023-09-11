
load('../data/abi_742951821_natscenes.mat')
trial_y_full=trial_y_full(:,mean(trial_y_full)>1);

xToZref = -1:max(trial_x_full);
xToZ = @(x) [ones(size(x)) double(xToZref==x)];
xGrid = [-1:max(trial_x_full)]';
N = size(trial_y_full,2);

% get cross-validated results
res = decode_cv_cat(trial_x_full,trial_y_full,xGrid,xToZ,10,1,100);

%% Fig 3C - tuning curve examples
zplot = xToZ(xGrid);
lam = exp(zplot*res.ols_res(1).Poi_BETA_ols);
figure(1)
ridx = [24 15 21 41 239];
for r=1:length(ridx)
    subplot(1,5,r)
    stairs(lam(:,ridx(r)))
    set(gca,'TickDir','out')
    box off
    axis tight
end

%% Fig 3F - example and average posteriors

figure(2)
clf
ridx = [10 11 14 27];
imsub = unique([trial_x_full(ridx)+2; [1:14]']);
for i=1:length(ridx)
    subplot(length(ridx),1,i)
    bar(res.ols_res_all.Poippp(imsub,ridx(i)),1,'EdgeColor','flat','FaceColor','b')
    hold on
    bar(res.lat_res_all.NBppp(imsub,ridx(i)),1,'EdgeColor','flat','FaceColor','r')

    plot(find(imsub==trial_x_full(ridx(i))+2,1),0,'^')
    hold off
    set(gca,'TickDir','out')
    box off
end

palign=[];
for i=1:length(trial_x_full)
    palign(:,i,1) = sort(res.ols_res_all.Poippp(:,i),'descend');
    palign(:,i,2) = sort(res.lat_res_all.NBppp(:,i),'descend');
end

figure(3)
bar(mean(palign(:,:,1),2),1,'EdgeColor','flat','FaceColor','b')
hold on
bar(mean(palign(:,:,2),2),1,'EdgeColor','flat','FaceColor','r')
hold off
set(gca,'TickDir','out')
box off

%% Fig 4C

coverVec = linspace(0.001,0.999,256);
coverage(1) = eval_coverage(res.ols_res_all.Poippp,xGrid,trial_x_full,coverVec);
coverage(2) = eval_coverage(res.ols_res_all.NBppp,xGrid,trial_x_full,coverVec);
coverage(3) = eval_coverage(res.lat_res_all.Poippp,xGrid,trial_x_full,coverVec);
coverage(4) = eval_coverage(res.lat_res_all.NBppp,xGrid,trial_x_full,coverVec);

figure(4)
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
pall=res.ols_res_all.Poippp;
coverCost = @(c) sum(coverage_r(normlogP((1./(1+exp(-c)))*log(pall')),xGrid,trial_x_full,coverVec,1)-coverVec).^2;
c = fminsearch(coverCost,-1);
cxpall_pdglm = normlogP((1./(1+exp(-c)))*log(pall'));
call(1)=c;

pall=res.lat_res_all.Poippp;
coverCost = @(c) sum(coverage_r(normlogP((1./(1+exp(-c)))*log(pall')),xGrid,trial_x_full,coverVec,1)-coverVec).^2;
c = fminsearch(coverCost,-1);
cxpall_pdfa = normlogP((1./(1+exp(-c)))*log(pall'));
call(2)=c;

pall=res.ols_res_all.NBppp;
coverCost = @(c) sum(coverage_r(normlogP((1./(1+exp(-c)))*log(pall')),xGrid,trial_x_full,coverVec,1)-coverVec).^2;
c = fminsearch(coverCost,-1);
cxpall_nbdglm = normlogP((1./(1+exp(-c)))*log(pall'));
call(3)=c;

pall=res.lat_res_all.NBppp;
coverCost = @(c) sum(coverage_r(normlogP((1./(1+exp(-c)))*log(pall')),xGrid,trial_x_full,coverVec,1)-coverVec).^2;
c = fminsearch(coverCost,-1);
cxpall_nbdfa = normlogP((1./(1+exp(-c)))*log(pall'));
call(4)=c;


coverVec = linspace(0.001,0.999,256);
cxp_coverage(1) = eval_coverage(cxpall_pdglm,xGrid,trial_x_full,coverVec);
cxp_coverage(2) = eval_coverage(cxpall_nbdglm,xGrid,trial_x_full,coverVec);
cxp_coverage(3) = eval_coverage(cxpall_pdfa,xGrid,trial_x_full,coverVec);
cxp_coverage(4) = eval_coverage(cxpall_nbdfa,xGrid,trial_x_full,coverVec);

% Fig 7C
figure(5)
for i=1:4
    plot(cxp_coverage(i).credMass_vec,cxp_coverage(i).pCover_corr)
    hold on
end
hold off    
xlim([0 1])
ylim([0 1])
line(xlim(),xlim())
legend({'PGLM','NBGLM','PGLLVM','NBGLLVM'})
xlabel('Credible Set')
ylabel('Coverage')

%% Fig 9C

edges = linspace(0,log2(length(xGrid)),32);
figure(6)
clf

[~,mapi] = max(res.ols_res_all.Poippp);
ee = trial_x_full==xGrid(mapi);
[phat,ci]=binofit(sum(ee),length(ee))
hh = nansum(-res.ols_res_all.Poippp.*log2(res.ols_res_all.Poippp))';
subplot(2,1,1)
histogram(hh,edges,'EdgeColor','none')
box off; set(gca,'TickDir','out')
subplot(2,1,2)
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
subplot(2,1,1)
hold on
histogram(hh,edges,'EdgeColor','none')
box off; set(gca,'TickDir','out')
hold off
subplot(2,1,2)
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


[~,mapi] = max(res.ols_res_all.Poippp);
ee = trial_x_full==xGrid(mapi);
hh = nansum(-cxpall_pdglm.*log2(cxpall_pdglm))';
edges = linspace(0,log2(length(xGrid)),32);
subplot(2,1,1)
hold on
histogram(hh,edges,'EdgeColor','none')
box off; set(gca,'TickDir','out')
hold off
subplot(2,1,2)
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