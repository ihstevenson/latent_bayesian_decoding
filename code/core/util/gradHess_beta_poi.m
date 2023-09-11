function gradHess = gradHess_beta_poi(vecBeta, beta0,Q0,Y,X,m,A,Q,offset)

% to debug
% vecBeta = beta(:);
% beta0 = beta_true(:,1);
% Q0 = eye(length(beta0));
% Y = Y;
% X = X;
% m = zeros(length(beta0),1);
% A = eye(length(beta0));
% Q = eye(length(beta0))*1e-4;
% offset = offset;

T = size(Y, 2);

beta_all = reshape(vecBeta, [], T);
mu_tmp = exp(X*beta_all + offset);

hessup = repmat((Q\A)', 1, 1, T-1);
hessub = repmat(Q\A, 1, 1, T-1);
hessmed = repmat(zeros(size(beta_all, 1)),1,1,T);
for t = 1:T
    hess_logPos = -X'*diag(mu_tmp(:,t))*X;
    if (t==1)
        hessmed(:,:,t) = hess_logPos - inv(Q0)- A'*(Q\A);
    elseif (t == T)
        hessmed(:,:,t) = hess_logPos -inv(Q);
    else
        hessmed(:,:,t) = hess_logPos - inv(Q) - A'*(Q\A);
    end
end

gradHess{1} = X'*(Y - mu_tmp)...
    + [-Q0\(beta_all(:,1) - beta0)+...
    A'*(Q\(beta_all(:,2) - A*beta_all(:,1)-m)),...
    -Q\(beta_all(:,2:(T-1)) - A*beta_all(:,1:(T-2))-m)+...
    A'*(Q\(beta_all(:,3:T) - A*beta_all(:,2:(T-1))-m)),...
    -Q\(beta_all(:,T) - A*beta_all(:,T-1)-m)];
gradHess{1} = gradHess{1}(:);

gradHess{2} = blktridiag(hessmed,hessub,hessup);


end