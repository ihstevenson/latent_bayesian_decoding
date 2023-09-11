function [beta,W,mu_pred,mu_filt, W01] =...
    ppasmoo_nbdglm(beta0,W0,Ytmp,Xtmp,alpha,m,A,Q,offset)

% to debug
% beta0 = beta_true(:,1);
% W0 = eye(length(beta0));
% Y = Y;
% X = X;
% alpha = alpha_true;
% m = zeros(length(beta0),1);
% A = eye(length(beta0));
% Q = eye(length(beta0))*1e-4;
% offset = offset;


% beta0 = beta0;
% W0 = Q0;
% Ytmp = Ytmp;
% Xtmp = Xtmp;
% alpha = alpha;
% m = m;
% A = A;
% Q = Q;
% offset = offset;




N = size(Ytmp, 1);
T = size(Ytmp, 2);

% Preallocate
beta = zeros(length(beta0), T);
W = zeros([size(W0) T]);
mu_pred = zeros(N,T);

% Initialize
beta(:,1)   = beta0;
W(:,:,1) = W0;
mu_pred(:,1) = exp(Xtmp*beta(:,1) + offset(:,1));

mu_filt = mu_pred;
betapred = beta;
Wpred = W;

% warning('Message 1.')
% Forward-Pass (Filtering)
for i=2:T
    betapred(:,i) = m + A*beta(:,i-1);
    Wpred(:,:,i) = A*W(:,:,i-1)*A' + Q;
    mu_pred(:,i) = exp(Xtmp*betapred(:, i) + offset(:,i));
    
    obsIdx = ~isnan(Ytmp(:,i));
    SCORE = Xtmp(obsIdx,:)'*((Ytmp(obsIdx,i) - mu_pred(obsIdx,i))./...
        (1 + alpha(obsIdx).*mu_pred(obsIdx,i)));
    INFO = Xtmp(obsIdx,:)'*diag((mu_pred(obsIdx,i).*(1 + alpha(obsIdx).*Ytmp(obsIdx,i)))./...
        ((1+alpha(obsIdx).*mu_pred(obsIdx,i)).^2))*Xtmp(obsIdx,:);
    
    Wpostinv = inv(Wpred(:,:,i)) + INFO;
    W(:,:,i) = inv(Wpostinv);
    beta(:,i)  = betapred(:,i) + W(:,:,i)*SCORE;
    
    mu_filt(:,i) = exp(Xtmp*beta(:,i) + offset(:,i));
    
    [~, msgid] = lastwarn;
    if strcmp(msgid,'MATLAB:nearlySingularMatrix') || strcmp(msgid,'MATLAB:illConditionedMatrix')
        lastwarn('')
        error('singular')
        lastwarn('')
        return;
    end
end

% W_filt = W;

lastwarn('')
I = eye(length(beta0));

W01 = zeros([size(W0) T-1]);
for i=(T-1):-1:1
    Wi = inv(Wpred(:,:,i+1));
    Fsquig = inv(A)*(I-Q*Wi);
    Ksquig = inv(A)*Q*Wi;
    
    beta(:,i)=Fsquig*beta(:,i+1) + Ksquig*betapred(:,i+1);
    C = W(:,:,i)*A'*Wi;
    W(:,:,i) = W(:,:,i) + C*(W(:,:,i+1)-Wpred(:,:,i+1))*C';
    W(:,:,i) = (W(:,:,i) + W(:,:,i)')/2;
    W01(:,:,i) = C*W(:,:,i+1);
    W01(:,:,i) = (W01(:,:,i) + W01(:,:,i)')/2;
end

% W02 = zeros([size(W0) T-1]);
% for i=(T-1):-1:1
%     Wi = inv(Wpred(:,:,i+1));
%     Cm1 = W_filt(:,:,i)*A'*Wi;
%     if i== (T-1)
%         W02(:,:,i) = W(:,:,i+1)*Cm1';
%     else
%         C = W_filt(:,:,i+1)*A'*inv(Wpred(:,:,i+2));
%         W02(:,:,i) = W_filt(:,:,i+1)*Cm1 + C*(W02(:,:,i+1)-A*W_filt(:,:,i+1))*Cm1';
%     end
% end
% 
% [squeeze(W01) squeeze(W02)]



end
