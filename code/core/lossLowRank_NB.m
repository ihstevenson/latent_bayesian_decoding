
function [f,dx] = lossLowRank_NB(b,y,k,alpha,offset,nu)

% if nargin<5, offset=0; end
if nargin<6, nu=0; end

% to debug
% b= [A_fit{ii-1}(:); B_fit{ii-1}(:)];
% y = Y;
% k = k;
% alpha = alpha_fit(:,ii);
% offset = OFFSET;
% nu = 0;
[n,m]=size(y);

A = reshape(b(1:(n*k)),n,k);
B = reshape(b((n*k+1):end),k,m);

xb = A*B + offset;

mu = exp(xb);
alpha_tmp = repmat(alpha', n, 1);
amu = alpha_tmp.*mu;
amup1 = 1 + amu;
Llhd = y.*log(amu./amup1) - (1./alpha_tmp).*log(amup1) +...
    gammaln(y + 1./alpha_tmp) - gammaln(y + 1) - gammaln(1./alpha_tmp);
% Llhd(~isfinite(Llhd)) = 0;

f = -nansum(Llhd, 'all');
mu_err = mu-y;
mu_err(~isfinite(mu_err))=0;

dA = (mu_err)*B'+nu*sign(A);
dB = A'*(mu_err)+nu*sign(B);
dx = [dA(:); dB(:)];

