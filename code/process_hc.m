
load('ec014_468_preprocessed_5Hz.mat')
cam2cm = 250/range(prctile(position_realigned(:,1),[1 99])); % unit conversion

p=20;
b = getCubicBSplineBasis(position_circular*pi,p,true);

xToZ = @(x) [ones(size(x)) getCubicBSplineBasis(x,20,true)];

x_encode = position_circular(3000:10000)*pi;
y_encode = spike_counts(3000:10000,:);
y_encode = y_encode(:,mean(y_encode)>.05);

ridx = randperm(size(y_encode,2));
% ridx = ridx(1:30);
y_encode = y_encode(:,ridx);
y_decode = y_encode;

z_encode = xToZ(x_encode);
xGrid = linspace(0,2*pi-2*pi/512, 512)';
zGrid = xToZ(xGrid);

T = size(y_encode,1);

%% encoding model: PDGLM
stats = glmMod(y_encode,z_encode,"Poisson", 'reg', 10); 
output_pdglm.Poi_BETA_ols = stats.BETA;
output_pdglm.place_fields = exp(zGrid*output_pdglm.Poi_BETA_ols);
output_pdglm.lambda = exp(z_encode*output_pdglm.Poi_BETA_ols);

%% decoding PDGLM

offset = repmat(output_pdglm.Poi_BETA_ols(1,:)',1,T);
B = output_pdglm.Poi_BETA_ols(2:end,:);

x0 = zeros(p,1);
Qx0 = eye(length(x0));
mx = zeros(p,1);
Ax = eye(p);
Qx = eye(p)*1e-3;
output_pdglm_decode = dynamicPGLM_EM(y_decode', B', offset, true,...
    'beta0', x0, 'Q0', Qx0, 'm',mx,'A',Ax,'Q',Qx);

%% evaluate along grid...
pall = zeros(size(zGrid,1),T);
for i=1:T
    pall(:,i) = mvnpdf(zGrid(:,2:end),output_pdglm_decode.beta(:,i)',output_pdglm_decode.W(:,:,i));
end
output_pdglm_decode.pall = pall./sum(pall);
[~,i] = max(output_pdglm_decode.pall);
output_pdglm_decode.map = xGrid(i);




%% encoding model: NBDGLM
stats = glmMod(y_encode,z_encode,"NB", 'reg', 10);
output_nbdglm.NB_BETA_ols = stats.BETA;
output_nbdglm.NB_alpha_ols = stats.ALPHA(1,:)';

%% decoding NBDGLM

offset = repmat(output_nbdglm.NB_BETA_ols(1,:)',1,T);
B = output_nbdglm.NB_BETA_ols(2:end,:);

x0 = zeros(p,1);
Qx0 = eye(length(x0));
mx = zeros(p,1);
Ax = eye(p);
Qx = eye(p)*1e-3;
output_nbdglm_decode = dynamicNBGLM_EM(y_decode', B', offset, output_nbdglm.NB_alpha_ols, true,...
    'beta0', x0, 'Q0', Qx0, 'm',mx,'A',Ax,'Q',Qx);

%% evaluate along grid...
pall = zeros(size(zGrid,1),T);
for i=1:T
    pall(:,i) = mvnpdf(zGrid(:,2:end),output_nbdglm_decode.beta(:,i)',output_nbdglm_decode.W(:,:,i));
end
output_nbdglm_decode.pall = pall./sum(pall);
[~,i] = max(output_nbdglm_decode.pall);
output_nbdglm_decode.map = xGrid(i);




%% encoding model PFA
output_pdfa = dynamicPFA_EM_ian(y_encode',z_encode',2,true);
Qx = cov(diff(z_encode(:,2:end)));
Qx = diag(diag(Qx));

%% decoding PFA

xc0 = [zeros(p,1);output_pdfa.c0];
Q0 = blkdiag(eye(p), output_pdfa.Q0);
mxc = [zeros(p,1); output_pdfa.mc];
Axc = blkdiag(eye(p), output_pdfa.Ac);
qxc = blkdiag(Qx, output_pdfa.Qc);
offset = repmat(output_pdfa.BETA(:,1),1,T);

output_pdfa_decode = dynamicPGLM_EM_ind_ian(y_decode',[output_pdfa.BETA(:,2:end) output_pdfa.D],p, offset,true,...
    'beta0', xc0, 'Q0', Q0, 'm',mxc,'A',Axc,'Q',qxc); % independent version

%% evaluate along grid...
pall = zeros(size(zGrid,1),T);
for i=1:T
    pall(:,i) = mvnpdf(zGrid(:,2:end),output_pdfa_decode.beta(1:p,i)',output_pdfa_decode.W(1:p,1:p,i));
end
output_pdfa_decode.pall = pall./sum(pall);
[~,i] = max(output_pdfa_decode.pall);
output_pdfa_decode.map = xGrid(i);




%% encoding model NBFA
output_nbdfa = dynamicNBFA_EM_ian(y_encode',z_encode',2,true);
Qx = cov(diff(z_encode(:,2:end)));
Qx = diag(diag(Qx));

%% decoding NBFA

xc0 = [zeros(p,1);output_pdfa.c0];
Q0 = blkdiag(eye(p), output_pdfa.Q0);
mxc = [zeros(p,1); output_pdfa.mc];
Axc = blkdiag(eye(p), output_pdfa.Ac);
qxc = blkdiag(Qx, output_pdfa.Qc);
offset = repmat(output_pdfa.BETA(:,1),1,T);

output_pdfa_decode = dynamicNBGLM_EM_ind_ian(y_decode',[output_nbdfa.BETA(:,2:end) output_nbdfa.D],p, offset, output_nbfa.ALPHA,true,...
    'beta0', xc0, 'Q0', Q0, 'm',mxc,'A',Axc,'Q',qxc); % independent version

%% evaluate along grid...
pall = zeros(size(zGrid,1),T);
for i=1:T
    pall(:,i) = mvnpdf(zGrid(:,2:end),output_nbdfa_decode.beta(1:p,i)',output_nbdfa_decode.W(1:p,1:p,i));
end
output_nbdfa_decode.pall = pall./sum(pall);
[~,i] = max(output_nbdfa_decode.pall);
output_nbdfa_decode.map = xGrid(i);



%%
figure(5)
plot(position_circular(3000:10000,1)*pi)
hold on
plot(output_pdfa_decode.map)
hold off
output_pdglm_decode.err = circ_dist(position_circular(3000:10000,1)*pi,output_pdglm_decode.map);
output_nbdglm_decode.err = circ_dist(position_circular(3000:10000,1)*pi,output_nbdglm_decode.map);
output_pdfa_decode.err = circ_dist(position_circular(3000:10000,1)*pi,output_pdfa_decode.map);
[median(abs(output_pdglm_decode.err)) median(abs(output_nbdglm_decode.err)) median(abs(output_pdfa_decode.err))]/pi*250 % error in cm


%%
xl = [0 1000];
figure(4)
subplot(4,1,1)
imagesc(output_pdglm_decode.pall(1:256,:)+flipud(output_pdglm_decode.pall(257:end,:)))
hold on
plot(position_realigned(3000:10000,1)*256,'r')
hold off
set(gca,'YDir','normal')
box off; set(gca,'TickDir','out')
xlim(xl)

subplot(4,1,2)
imagesc(output_nbdglm_decode.pall(1:256,:)+flipud(output_nbdglm_decode.pall(257:end,:)))
hold on
plot(position_realigned(3000:10000,1)*256,'r')
hold off
set(gca,'YDir','normal')
box off; set(gca,'TickDir','out')
xlim(xl)

subplot(4,1,3)
imagesc(output_pdfa_decode.pall(1:256,:)+flipud(output_pdfa_decode.pall(257:end,:)))
hold on
plot(position_realigned(3000:10000,1)*256,'r')
hold off
set(gca,'YDir','normal')
box off; set(gca,'TickDir','out')
xlim(xl)

entr_pdglm = -sum(output_pdglm_decode.pall.*log2(output_pdglm_decode.pall+(output_pdglm_decode.pall==0)));
entr_nbdglm = -sum(output_nbdglm_decode.pall.*log2(output_nbdglm_decode.pall+(output_nbdglm_decode.pall==0)));
entr_pdfa = -sum(output_pdfa_decode.pall.*log2(output_pdfa_decode.pall+(output_pdfa_decode.pall==0)));
subplot(4,1,4)
plot(entr_pdglm)
hold on
plot(entr_nbdglm)
plot(entr_pdfa)
hold off
box off; set(gca,'TickDir','out')
xlim(xl)

[nanmean(entr_pdglm) nanmean(entr_nbdglm) nanmean(entr_pdfa)]