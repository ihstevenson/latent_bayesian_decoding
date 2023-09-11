function output = dynamicPGLM_EM_ind(Ytmp,Xtmp,p1,offset,verbose,varargin)

% to debug
% Ytmp = Y;
% Xtmp = D_fit{ii-1};
% offset = offset;


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
                tol_init = varargin{c+1};
            case {'maxIter'}
                maxIter_init = varargin{c+1};
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

mx = m(1:p1);
mc = m((p1+1):end);
Ax = A(1:p1,1:p1);
Ac = A((p1+1):end,(p1+1):end);

T = size(Ytmp,2);
LLHD = zeros(maxIter, 1);

for ii = 1:maxIter
    
    [beta,W,~,~, W01] =...
        ppasmoo_poidglm(beta0,Q0,Ytmp,Xtmp,m,A,Q,offset);
    gradHess = @(vecBeta) gradHess_beta_poi(vecBeta, beta0,Q0,Ytmp,Xtmp,m,A,Q,offset);
    [vecC,~,~,~, ~] = newtonGH(gradHess,beta(:),1e-6,1000,false);
    beta_fit{ii} = reshape(vecC,[], T);
    
    
    % M-step: update {x0,Qx0,mx,Ax,Qx} & {c0,Qc0,mc,Ac,Qc}
    mu11_x = zeros(p1,p1,T); % E(c_{t}*c'_{t})
    mu10_x = zeros(p1,p1,T-1); % E(c_{t}c'_{t-1})
    for tt=1:T
       mu11_x(:,:,tt) =  W(1:p1,1:p1,tt) + beta_fit{ii}(1:p1,tt)*(beta_fit{ii}(1:p1,tt))';
       if tt>=2
           mu10_x(:,:,tt-1) = W01(1:p1,1:p1,tt-1) + beta_fit{ii}(1:p1,tt)*(beta_fit{ii}(1:p1,tt-1))';
       end
    end
    
    mu11_c = zeros(ptmp-p1,ptmp-p1,T); % E(c_{t}*c'_{t})
    mu10_c = zeros(ptmp-p1,ptmp-p1,T-1); % E(c_{t}c'_{t-1})
    for tt=1:T
       mu11_c(:,:,tt) =  W((p1+1):end,(p1+1):end,tt) + beta_fit{ii}((p1+1):end,tt)*(beta_fit{ii}((p1+1):end,tt))';
       if tt>=2
           mu10_c(:,:,tt-1) = W01((p1+1):end,(p1+1):end,tt-1) + beta_fit{ii}((p1+1):end,tt)*(beta_fit{ii}((p1+1):end,tt-1))';
       end
    end
    
    % a. beta0 = (x0,c0) ,Q0 = (Qx0 & Qc0)
    beta0 = beta_fit{ii}(:,1);
    Q0 = blkdiag(W(1:p1,1:p1,1),W((p1+1):end,(p1+1):end,1));
    
    % b. update mx, mc, Ax, Ac
    smu1_x = sum(beta_fit{ii}(1:p1,2:T),2);
    smu10_x = sum(mu10_x,3);
    smu0_x = sum(beta_fit{ii}(1:p1,1:(T-1)),2);
    smu00_x = sum(mu11_x(:,:,1:(T-1)), 3);
    smu11_x = sum(mu11_x(:,:,2:T), 3);
    mA_tmp_x = [smu1_x smu10_x]/([T-1 smu0_x';smu0_x smu00_x]);
%     mx = mA_tmp_x(:,1);
%     Ax = mA_tmp_x(:,2:end);
    
    smu1_c = sum(beta_fit{ii}((p1+1):end,2:T),2);
    smu10_c = sum(mu10_c,3);
    smu0_c = sum(beta_fit{ii}((p1+1):end,1:(T-1)),2);
    smu00_c = sum(mu11_c(:,:,1:(T-1)), 3);
    smu11_c = sum(mu11_c(:,:,2:T), 3);
    mA_tmp_c = [smu1_c smu10_c]/([T-1 smu0_c';smu0_c smu00_c]);
%     mc = mA_tmp_c(:,1);
%     Ac = mA_tmp_c(:,2:end);
    
%     m = [mx;mc];
%     A = blkdiag(Ax, Ac);
    
    % c. Qx & Qc
    Qx = (smu11_x - mx*smu1_x' - Ax*smu10_x' - smu1_x*mx' - smu10_x*Ax' +...
        mx*mx' + Ax*smu0_x*mx' + mx*smu0_x'*Ax' + Ax*smu00_x*Ax')/(T-1);
    Qc = (smu11_c - mc*smu1_c' - Ac*smu10_c' - smu1_c*mc' - smu10_c*Ac' +...
        mc*mc' + Ac*smu0_c*mc' + mc*smu0_c'*Ac' + Ac*smu00_c*Ac')/(T-1);
    
    Q = blkdiag(Qx,Qc);
    
    % (5) evaluate the llhd/spk
    MU_tmp = exp(Xtmp*beta_fit{ii});
    LLHD(ii) = nansum(-MU_tmp + Ytmp.*log(MU_tmp + (MU_tmp == 0)), 'all')/ nansum(Ytmp,'all');
    
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