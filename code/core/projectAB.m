function [A_out,B_out, d_out] = projectAB(A,B,k,ver)

% to debug
% A = Aout;
% B = Bout;
% k = k;
% ver = ProjVer;

% [U,D,V] = svd(A*B, 'econ');
% A_new = U(:,1:k);
% D_new = D(1:k, 1:k);
% V_new = V(:,1:k);

[U_new,D_new,V_new] = svds(A*B, k);

% permute
[~,DIdx] = sort(abs(diag(D_new)), 'descend');
U_new = U_new(:,DIdx);
D_new = D_new(DIdx,DIdx);
V_new = V_new(:,DIdx);

% sign: in practice, rarely 0 (just use 1st row)
signU = sign(U_new(1,:))';
singD = sign(diag(D_new));
U_new = U_new*diag(signU);
D_new = abs(D_new);
V_new = V_new*diag(singD.*signU);

if ver == 1
    
    % version 1: A large, B small
    A_out = U_new*D_new;
    B_out = V_new';
    d_out = diag(D_new);
elseif ver == 2
    
    % version 2: A small, B large
    A_out = U_new;
    B_out = D_new*V_new';
    d_out = diag(D_new);
else
    error('ver should = 1 or 2');
end







end