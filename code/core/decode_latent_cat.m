% Bayesian decoding with Poisson anb NB GLLVMs (categorical data)
% Inputs:
%  z_encode [trials x p] matrix of covariates to fit encoding model
%  y_encode [trials x N] matrix of spike count observations to fit encoding model
%  x_decode [test_trials x 1] vector of true decoding values (only used for error calculations)
%  y_decode [test_trials x N] matrix of spike count observations to decode
%  k - number of latent dimensions to use
%  xGrid - vector for decoding grid
%  xToZ - function to represent x as a basis
%  reg (optional) - regularization parameter for the tuning coefficients
%
% Output:
%   struct with results for Poisson GLM and NB GLLVM

function res = decode_latent_cat(z_encode,y_encode,x_decode,y_decode,k,xGrid,xToZ,reg)

if nargin<8
    reg = 10e-6;
end
tol_fit = 1e-6;

fprintf('encode_latent poiss...\n')

% model 1: Poisson latent
res.PoiMod = latFacMod(y_encode,z_encode,k,"Poisson",...
            "minFunc", 2, true, 'tol', tol_fit, 'reg', reg, 'maxIter', 100);

fprintf('encode_latent nb...\n')

% model 2: NB latent
res.NBMod = latFacMod(y_encode,z_encode,k,"NB",...
            "minFunc", 2, true, 'tol', tol_fit, 'reg', reg, 'maxIter', 100);

res.Poippp = zeros(length(xGrid),length(x_decode));
res.NBppp = zeros(length(xGrid),length(x_decode));
res.Poimap = zeros(1, length(x_decode));
res.NBmap = zeros(1, length(x_decode));

fprintf('decode_latent poiss...\n')

zGrid = xToZ(xGrid);
for aa = 1:length(x_decode)
    y_star = y_decode(aa,:)';
    
    % (1) Poisson
    logMar = zeros(length(xGrid),1);
    offset = zGrid*res.PoiMod.BETA;
    
    for tt = 1:length(xGrid)
        logMar(tt) = logMarA_poi(offset(tt,:)', y_star, res.PoiMod.B);
    end

    c=max(logMar);
    logzall=(c+log(sum(exp(logMar-c))));
    marLl = exp(logMar-logzall);
    res.Poippp(:,aa)=marLl;
    [~,res.Poimap(aa)] = max(marLl);
end

fprintf('decode_latent nb...\n')
for aa = 1:length(x_decode)
    y_star = y_decode(aa,:)';
    
    % (2) NB
    logMar = zeros(length(xGrid),1);
    offset = zGrid*res.NBMod.BETA;
    for tt = 1:length(xGrid)
        logMar(tt) = logMarA_NB(offset(tt,:)', y_star, res.NBMod.B, res.NBMod.alpha);
    end

    c=max(logMar);
    logzall=(c+log(sum(exp(logMar-c))));
    marLl = exp(logMar-logzall);
    res.NBppp(:,aa)=marLl;
    [~,res.NBmap(aa)] = max(marLl);
end
