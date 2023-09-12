
load('../data/ec014_468_preprocessed_5Hz.mat')
cam2cm = 250/range(prctile(position_realigned(:,1),[1 99])); % unit conversion
time_range = 3000:10000;

p=20;
b = getCubicBSplineBasis(position_circular*pi,p,true);

xToZ = @(x) [ones(size(x)) getCubicBSplineBasis(x,20,true)];

x_encode = position_circular(time_range)*pi;
y_encode = spike_counts(time_range,:);
y_encode = y_encode(:,mean(y_encode)>.05);
x_decode = position_circular(time_range)*pi;
y_decode = y_encode; % (don't cross-validate) just apply to decoder to training

z_encode = xToZ(x_encode);
xGrid = linspace(0,2*pi-2*pi/512, 512)';
zGrid = xToZ(xGrid);

T = size(y_encode,1);


%% Poisson Dynamic GLM

% encoding model: PDGLM
stats = glmMod(y_encode,z_encode,"Poisson", 'reg', 10); 
output_pdglm.Poi_BETA_ols = stats.BETA;
output_pdglm.place_fields = exp(zGrid*output_pdglm.Poi_BETA_ols);
output_pdglm.lambda = exp(z_encode*output_pdglm.Poi_BETA_ols);

% decoding PDGLM
offset = repmat(output_pdglm.Poi_BETA_ols(1,:)',1,T);
B = output_pdglm.Poi_BETA_ols(2:end,:);
x0 = zeros(p,1);
Qx0 = eye(length(x0));
mx = zeros(p,1);
Ax = eye(p);
Qx = eye(p)*1e-3;

output_pdglm_decode = dynamicPGLM_EM(y_decode', B', offset, true,...
    'beta0', x0, 'Q0', Qx0, 'm',mx,'A',Ax,'Q',Qx);

% evaluate the Laplace approximation along grid...
pall = zeros(size(zGrid,1),T);
for i=1:T
    pall(:,i) = mvnpdf(zGrid(:,2:end),output_pdglm_decode.beta(:,i)',output_pdglm_decode.W(:,:,i));
end
output_pdglm_decode.pall = pall./sum(pall);
[~,i] = max(output_pdglm_decode.pall);
output_pdglm_decode.map = xGrid(i);



%% Negative Binomial Dynamic GLM

% encoding model: NBDGLM
stats = glmMod(y_encode,z_encode,"NB", 'reg', 10);
output_nbdglm.NB_BETA_ols = stats.BETA;
output_nbdglm.NB_alpha_ols = stats.ALPHA(1,:)';

% decoding NBDGLM
offset = repmat(output_nbdglm.NB_BETA_ols(1,:)',1,T);
B = output_nbdglm.NB_BETA_ols(2:end,:);
x0 = zeros(p,1);
Qx0 = eye(length(x0));
mx = zeros(p,1);
Ax = eye(p);
Qx = eye(p)*1e-3;

output_nbdglm_decode = dynamicNBGLM_EM(y_decode', B', offset, output_nbdglm.NB_alpha_ols, true,...
    'beta0', x0, 'Q0', Qx0, 'm',mx,'A',Ax,'Q',Qx);

% evaluate the Laplace approximation along grid...
pall = zeros(size(zGrid,1),T);
for i=1:T
    pall(:,i) = mvnpdf(zGrid(:,2:end),output_nbdglm_decode.beta(:,i)',output_nbdglm_decode.W(:,:,i));
end
output_nbdglm_decode.pall = pall./sum(pall);
[~,i] = max(output_nbdglm_decode.pall);
output_nbdglm_decode.map = xGrid(i);



%% Poisson Dynamic GLLVM (PDFA)

% encoding model PFA
output_pdfa = dynamicPFA_EM(y_encode',z_encode',1,true,'reg',10);
Qx = cov(diff(z_encode(:,2:end)));
Qx = diag(diag(Qx));

% decoding PFA
xc0 = [zeros(p,1);output_pdfa.c0];
Q0 = blkdiag(eye(p), output_pdfa.Q0);
mxc = [zeros(p,1); output_pdfa.mc];
Axc = blkdiag(eye(p), output_pdfa.Ac);
qxc = blkdiag(Qx, output_pdfa.Qc);
offset = repmat(output_pdfa.BETA(:,1),1,T);

output_pdfa_decode = dynamicPGLM_EM_ind(y_decode',[output_pdfa.BETA(:,2:end) output_pdfa.D],p, offset,true,...
    'beta0', xc0, 'Q0', Q0, 'm',mxc,'A',Axc,'Q',qxc); % independent version

% evaluate the Laplace approximation along grid...
pall = zeros(size(zGrid,1),T);
for i=1:T
    pall(:,i) = mvnpdf(zGrid(:,2:end),output_pdfa_decode.beta(1:p,i)',output_pdfa_decode.W(1:p,1:p,i));
end
output_pdfa_decode.pall = pall./sum(pall);
[~,i] = max(output_pdfa_decode.pall);
output_pdfa_decode.map = xGrid(i);



%% Negative Binomial Dynamic GLLVM (NBDFA)

% encoding model NBFA
output_nbdfa = dynamicNBFA_EM(y_encode',z_encode',1,true);
Qx = cov(diff(z_encode(:,2:end)));
Qx = diag(diag(Qx));

% decoding NBFA
xc0 = [zeros(p,1); output_nbdfa.c0];
Q0 = blkdiag(eye(p), output_nbdfa.Q0)*10e-3;
mxc = [zeros(p,1); output_nbdfa.mc];
Axc = blkdiag(eye(p), output_nbdfa.Ac);
qxc = blkdiag(Qx, output_nbdfa.Qc);
offset = repmat(output_nbdfa.BETA(:,1),1,T);

output_nbdfa_decode = dynamicNBGLM_EM_ind(y_decode', [output_nbdfa.BETA(:,2:end) output_nbdfa.D],p, offset, output_nbdfa.alpha, true,...
    'beta0', xc0, 'Q0', Q0, 'm',mxc,'A',Axc,'Q',qxc);

% evaluate the Laplace approximation along grid...
pall = zeros(size(zGrid,1),T);
for i=1:T
    pall(:,i) = mvnpdf(zGrid(:,2:end),output_nbdfa_decode.beta(1:p,i)',output_nbdfa_decode.W(1:p,1:p,i));
end
output_nbdfa_decode.pall = pall./sum(pall);
[~,i] = max(output_nbdfa_decode.pall);
output_nbdfa_decode.map = xGrid(i);


%% Fig 8B

coverVec = linspace(0.001,0.999,256);
coverage(1) = eval_coverage(output_pdglm_decode.pall,xGrid,x_decode,coverVec);
coverage(2) = eval_coverage(output_nbdglm_decode.pall,xGrid,x_decode,coverVec);
coverage(3) = eval_coverage(output_pdfa_decode.pall,xGrid,x_decode,coverVec);
coverage(4) = eval_coverage(output_nbdfa_decode.pall,xGrid,x_decode,coverVec);

figure(21)
clf
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
pall=output_pdglm_decode.pall;
coverCost = @(c) sum(coverage_r(normlogP((1./(1+exp(-c)))*log(pall')),xGrid,x_decode,coverVec)-coverVec).^2;
c = fminsearch(coverCost,-1);
cxpall_pdglm = normlogP((1./(1+exp(-c)))*log(pall'));
call(1)=c;

pall=output_nbdglm_decode.pall;
coverCost = @(c) sum(coverage_r(normlogP((1./(1+exp(-c)))*log(pall')),xGrid,x_decode,coverVec)-coverVec).^2;
c = fminsearch(coverCost,-1);
cxpall_nbdglm = normlogP((1./(1+exp(-c)))*log(pall'));
call(2)=c;

pall=output_pdfa_decode.pall;
coverCost = @(c) sum(coverage_r(normlogP((1./(1+exp(-c)))*log(pall')),xGrid,x_decode,coverVec)-coverVec).^2;
c = fminsearch(coverCost,-1);
cxpall_pdfa = normlogP((1./(1+exp(-c)))*log(pall'));
call(3)=c;

pall=output_nbdfa_decode.pall;
coverCost = @(c) sum(coverage_r(normlogP((1./(1+exp(-c)))*log(pall')),xGrid,x_decode,coverVec)-coverVec).^2;
c = fminsearch(coverCost,-1);
cxpall_nbdfa = normlogP((1./(1+exp(-c)))*log(pall'));
call(4)=c;


coverVec = linspace(0.001,0.999,256);
cxp_coverage(1) = eval_coverage(cxpall_pdglm,xGrid,x_decode,coverVec);
cxp_coverage(2) = eval_coverage(cxpall_nbdglm,xGrid,x_decode,coverVec);
cxp_coverage(3) = eval_coverage(cxpall_pdfa,xGrid,x_decode,coverVec);
cxp_coverage(4) = eval_coverage(cxpall_nbdfa,xGrid,x_decode,coverVec);

% Fig 8C
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

%% Figure 8A (visualize segment)
% note: direction is decoded, but ignored in these plots

xl = [1800 2200]; % shown in Fig
xl = [1000 3000];
figure(4)
subplot(3,1,1)
imagesc(output_pdglm_decode.pall(1:256,:)+flipud(output_pdglm_decode.pall(257:end,:)))
hold on
plot(position_realigned(time_range,1)*256,'r')
hold off
set(gca,'YDir','normal')
box off; set(gca,'TickDir','out')
xlim(xl)

subplot(3,1,2)
imagesc(output_nbdfa_decode.pall(1:256,:)+flipud(output_nbdfa_decode.pall(257:end,:)))
hold on
plot(position_realigned(time_range,1)*256,'r')
hold off
set(gca,'YDir','normal')
box off; set(gca,'TickDir','out')
xlim(xl)

subplot(3,1,3)
% imagesc(output_nbdfa_decode.pall(1:256,:)+flipud(output_nbdfa_decode.pall(257:end,:)))
imagesc(cxpall_nbdfa(1:256,:)+flipud(cxpall_nbdfa(257:end,:)))
hold on
plot(position_realigned(time_range,1)*256,'r')
hold off
set(gca,'YDir','normal')
box off; set(gca,'TickDir','out')
xlim(xl)