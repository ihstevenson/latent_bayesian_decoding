
function [f,dx] = lossLowRank_poi(b,y,k,offset,nu)

% nu = 0;
% if nargin<4, offset=0; end
if nargin<5, nu=0; end

% to debug
% b= [A_fit{ii-1}(:); B_fit{ii-1}(:)];
% y = Y;
% k = k;
% offset = OFFSET;
% nu = 0;
[n,m]=size(y);

A = reshape(b(1:(n*k)),n,k);
B = reshape(b((n*k+1):end),k,m);

xb = A*B + offset;
lam = exp(xb);

f = -nansum(nansum(y.*xb - lam)) + nu*sum(sum(abs(A))) + nu*sum(sum(abs(B)));
lam_err = lam-y;
lam_err(~isfinite(lam_err))=0;

dA = (lam_err)*B'+nu*sign(A);
dB = A'*(lam_err)+nu*sign(B);
dx = [dA(:); dB(:)];

