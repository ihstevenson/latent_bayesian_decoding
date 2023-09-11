function output = dynamicNBGLM_EM(Ytmp,Xtmp,offset,alpha,verbose,varargin)

tol = 1e-5;
maxIter = 1000;

ptmp = size(Xtmp,2);
beta0 = zeros(ptmp,1);
Q0 = eye(ptmp);
m = zeros(ptmp,1);
A = eye(ptmp);
Q = eye(ptmp)*1e-2;

if (~isempty(varargin))
    c = 1 ;
    while c <= length(varargin)
        switch varargin{c}
            case {'tol'}
                tol = varargin{c+1};
            case {'maxIter'}
                maxIter = varargin{c+1};
            case {'beta0'}
                beta0 = varargin{c+1};
            case {'Q0'}
                Q0 = varargin{c+1};
            case {'m'}
                m = varargin{c+1};
            case {'A'}
                A = varargin{c+1};
            case {'Q'}
                Q = varargin{c+1};
        end % switch
        c = c + 2;
    end % for
end % if

T = size(Ytmp,2);
LLHD = zeros(maxIter, 1);

for ii = 1:maxIter
    
    [beta,W,~,~, W01] =...
        ppasmoo_nbdglm(beta0,Q0,Ytmp,Xtmp,alpha,m,A,Q,offset);
    gradHess = @(vecBeta) gradHess_beta_nb(vecBeta, beta0,Q0,Ytmp,Xtmp,alpha,m,A,Q,offset);
    [vecC,~,~,~, ~] = newtonGH(gradHess,beta(:),1e-6,1000,false);
    beta_fit{ii} = reshape(vecC,[], T);
    
    
    % M-step: update c0, Q0, mc, Ac and Qc
    
    mu11 = zeros(ptmp,ptmp,T); % E(c_{t}*c'_{t})
    mu10 = zeros(ptmp,ptmp,T-1); % E(c_{t}c'_{t-1})
    for tt=1:T
       mu11(:,:,tt) =  W(:,:,tt) + beta_fit{ii}(:,tt)*(beta_fit{ii}(:,tt))';
       if tt>=2
           mu10(:,:,tt-1) = W01(:,:,tt-1) + beta_fit{ii}(:,tt)*(beta_fit{ii}(:,tt-1))';
       end
    end
    
    % a. beta0 & Q0
    beta0 = beta_fit{ii}(:,1);
    Q0 = W(:,:,1);
    
    % b. m & A
    smu1 = sum(beta_fit{ii}(:,2:T),2);
    smu10 = sum(mu10,3);
    smu0 = sum(beta_fit{ii}(:,1:(T-1)),2);
    smu00 = sum(mu11(:,:,1:(T-1)), 3);
    smu11 = sum(mu11(:,:,2:T), 3);
    
    mA_tmp = [smu1 smu10]/([T-1 smu0';smu0 smu00]);
    
    % c. Q
    Q = (smu11 - m*smu1' - A*smu10' - smu1*m' - smu10*A' +...
        m*m' + A*smu0*m' + m*smu0'*A' + A*smu00*A')/(T-1);
    
    % (5) evaluate the llhd/spk
    MU_tmp = exp(Xtmp*beta_fit{ii});
    ALPH_tmp = repmat(alpha, 1, T);
    AMU_tmp = ALPH_tmp.*MU_tmp;
    AMUp1_tmp = 1 + AMU_tmp;
    LLHD(ii) = nansum(Ytmp.*log(AMU_tmp./AMUp1_tmp) - (1./ALPH_tmp).*log(AMUp1_tmp) +...
        gammaln(Ytmp + 1./ALPH_tmp) - gammaln(Ytmp + 1) - gammaln(1./ALPH_tmp), 'all')/ nansum(Ytmp,'all');
    
    if ii > 1
        change = abs(LLHD(ii-1)/LLHD(ii) - 1);
        if verbose; disp("Iter" + ii + ": " + LLHD(ii) + ", Q(1) = " +...
                Q(1,1) + ", llhd_change: " + change); end
    end
    if ii > 10
       change_pre = abs(LLHD(ii-2)/LLHD(ii-1) - 1);
       if change > change_pre || change < tol
           break;
       end
    end
    
    
end

output.beta0 = beta0;
output.Q0 = Q0;
output.m = m;
output.A = A;
output.Q = Q;

output.beta = beta_fit{ii};
output.W = W;



end