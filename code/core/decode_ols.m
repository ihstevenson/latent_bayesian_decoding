% Bayesian decoding with Poisson anb NB GLMs
% Inputs:
%  z_encode [trials x p] matrix of covariates to fit encoding model
%  y_encode [trials x N] matrix of spike count observations to fit encoding model
%  x_decode [test_trials x 1] vector of true decoding values (only used for error calculations)
%  y_decode [test_trials x N] matrix of spike count observations to decode
%  xGrid - vector for decoding grid
%  xToZ - function to represent x as a basis
%  reg (optional) - regularization parameter for the tuning coefficients
%
% Output:
%   struct with results for Poisson GLM and NB GLM

function res = decode_ols(z_encode,y_encode,x_decode,y_decode,xGrid,xToZ, reg)


if nargin<7
    reg = 10e-6;
end

% encode
stats = glmMod(y_encode,z_encode,"Poisson", 'reg', reg); 
res.Poi_BETA_ols = stats.BETA;

stats = glmMod(y_encode,z_encode,"NB", 'reg', reg);
res.NB_BETA_ols = stats.BETA;
res.NB_alpha_ols = stats.ALPHA(1,:)';


% decode
lamx_sing = @(x) exp(xToZ(x)*res.Poi_BETA_ols);
mu_sing = @(x) exp(xToZ(x)*res.NB_BETA_ols);
alph = repmat(res.NB_alpha_ols', length(xGrid), 1);

res.lam = lamx_sing(xGrid);
res.mu = mu_sing(xGrid);

res.Poippp = zeros(length(xGrid),length(x_decode));
res.NBppp = zeros(length(xGrid),length(x_decode));
res.Poimap_ols = zeros(1, length(x_decode));
res.NBmap_ols = zeros(1, length(x_decode));

for aa = 1:length(x_decode)
    y_star = y_decode(aa,:)';
    y_star_expand = repmat(y_star',length(xGrid), 1);
    
    % (1) Poisson
    llhd = -lamx_sing(xGrid) + y_star'.*log(lamx_sing(xGrid) + (lamx_sing(xGrid) == 0));
    c=max(llhd);
    logz=(c+log(sum(exp(llhd-c))));
    p = exp(llhd-logz);

    logpall = sum(log(p),2);
    c=max(logpall);
    logzall=(c+log(sum(exp(logpall-c))));
    pall = exp(logpall-logzall);

    res.Poippp(:,aa)=pall;
    [~,res.Poimap_ols(aa)] = max(pall);

    % (2) NB
    mu = mu_sing(xGrid);
    amu = alph.*mu;
    amup1 = 1 + amu;
    llhd = y_star_expand.*log(amu./amup1) - (1./alph).*log(amup1) +...
        gammaln(y_star_expand + 1./alph) - gammaln(y_star_expand + 1) - gammaln(1./alph);
    c=max(llhd);
    logz=(c+log(sum(exp(llhd-c))));
    p = exp(llhd-logz);

    logpall = sum(log(p),2);
    c=max(logpall);
    logzall=(c+log(sum(exp(logpall-c))));
    pall = exp(logpall-logzall);

    res.NBppp(:,aa)=pall;
    [~,res.NBmap_ols(aa)] = max(pall);
end

xUse = ~isnan(res.Poimap_ols);
res.Poierr = circ_dist(x_decode(xUse),xGrid(res.Poimap_ols(xUse)));
res.PoiMed_ols = median(abs(res.Poierr))*180/pi;
res.PoiMean_ols = mean(abs(res.Poierr))*180/pi;

res.NBerr = circ_dist(x_decode(xUse),xGrid(res.NBmap_ols(xUse)));
res.NBMed_ols = median(abs(res.NBerr))*180/pi;
res.NBMean_ols = mean(abs(res.NBerr))*180/pi;
