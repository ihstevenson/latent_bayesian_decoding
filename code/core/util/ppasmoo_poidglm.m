function [beta,W,lamPred,lam, W01] = ppasmoo_poidglm(beta0,W0,Y,X,m,A,Q,offset)

% to debug
% beta0 = beta_true(:,1);
% W0 = eye(length(beta0));
% Y = Y;
% X = X;
% m = zeros(length(beta0),1);
% A = eye(length(beta0));
% Q = eye(length(beta0))*1e-4;
% offset = offset;

warning('off');
lastwarn('')
N = size(Y, 1);
T = size(Y, 2);

% Preallocate
beta   = zeros(length(beta0),T);
W   = zeros([size(W0) T]);
lamPred = zeros(N, T);

% Initialize
beta(:,1)   = beta0;
W(:,:,1) = W0;
lamPred(:,1)   = exp(X*beta0 + offset(:,1));

xpred = beta;
Wpred = W;
lam = lamPred;

I = eye(size(W0));

% Forward-Pass (Filtering)
for i=2:size(Y,2)
    xpred(:,i) = A*beta(:,i-1) + m;
    lamPred(:,i) = exp(X*xpred(:,i) + offset(:,i));
    Wpred(:,:,i) = A*W(:,:,i-1)*A' + Q;
    
    INFO = zeros(size(W0));
    SCORE = zeros(size(beta0));
    
    for k=1:N
        if(~isnan(Y(k,i)))
            INFO = INFO + X(k,:)'*(lamPred(k,i))*X(k,:);
            SCORE = SCORE + X(k,:)'*(Y(k,i) - lamPred(k, i));
        end
    end
    Wpostinv = inv(Wpred(:,:,i)) + INFO;
    W(:,:,i) = inv(Wpostinv);
    W(:,:,i) = (W(:,:,i) + W(:,:,i)')/2;
    
    beta(:,i)  = xpred(:,i) + W(:,:,i)*SCORE;
    
    lam(:,i) = exp(X*beta(:,i) + offset(:,i));
    
    [~, msgid] = lastwarn;
    if strcmp(msgid,'MATLAB:illConditionedMatrix')
        error('Error: singular matrix');
    end
end
% lastwarn('')


% Backward-Pass (RTS)
W01 = zeros([size(W0) T-1]);
for i=(T-1):-1:1
    Wi = inv(Wpred(:,:,i+1));
    J = W(:,:,i)*A'*Wi;
    beta(:,i) = beta(:,i) + J*(beta(:,i+1) - xpred(:,i+1));
    W(:,:,i) = W(:,:,i) + J*(W(:,:,i+1)-Wpred(:,:,i+1))*J';
    W(:,:,i) = (W(:,:,i) + W(:,:,i)')/2;
    W01(:,:,i) = J*W(:,:,i+1);
    W01(:,:,i) = (W01(:,:,i) + W01(:,:,i)')/2;
    
    lam(:,i) = exp(X*beta(:,i) + offset(:,i));
end






warning('on');


end