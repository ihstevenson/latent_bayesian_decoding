function output = dynamicPFA_EM(Y,X,q, verbose, varargin)


% to debug
% verbose = true;

tol_init = 1e-3;
maxIter_init = 500;
tol = 1e-5;
maxIter = 1000;
Q0 = eye(q);

mc = zeros(q,1);
Ac = eye(q);
Qc = eye(q)*1e-4;

if (~isempty(varargin))
    c = 1 ;
    while c <= length(varargin)
        switch varargin{c}
            case {'tol_init'}
                tol_init = varargin{c+1};
            case {'maxIter_init'}
                maxIter_init = varargin{c+1};
            case {'tol'}
                tol = varargin{c+1};
            case {'maxIter'}
                maxIter = varargin{c+1};
            case {'Q0'}
                Q0 = varargin{c+1};
            case {'mc'}
                mc = varargin{c+1};
            case {'Ac'}
                Ac = varargin{c+1};
            case {'Qc'}
                Qc = varargin{c+1};
        end % switch
        c = c + 2;
    end % for
end % if

N = size(Y,1);
T = size(Y,2);
p = size(X,1);

% A. initialization
mod_plat = latFacMod(Y',X',q,"Poisson","minFunc", 2, verbose,...
    'tol', tol_init, 'maxIter', maxIter_init, 'reg', 10);

% B. coordinate descent
BETA_fit{1} = mod_plat.BETA';
D_fit{1} = mod_plat.B';
C_fit{1} = mod_plat.A';

c0 = C_fit{1}(:,1);

LLHD = zeros(maxIter, 1);
LLHD(1) = mod_plat.LLHD;

for ii = 2:maxIter
    try
        % (1) update {ct}
        offset = BETA_fit{ii-1}*X;
        [C_tmp,W,~,~, W01] = ppasmoo_poidglm(c0,Q0,Y,D_fit{ii-1},...
            mc,Ac,Qc,offset);
        gradHess = @(vecBeta) gradHess_beta_poi(vecBeta, c0,Q0,Y,D_fit{ii-1},...
            mc,Ac,Qc,offset);
        [vecC,~,~,~, ~] = newtonGH(gradHess,C_tmp(:),1e-6,1000,false);
        C_fit{ii} = reshape(vecC,[], T);

        % (2) update BETA, D and alpha
        BETA_fit{ii} = zeros(size(BETA_fit{ii-1}));
        D_fit{ii} = zeros(size(D_fit{ii-1}));

        stats = glmMod(Y',[X' C_fit{ii}'],"Poisson", 'reg', 10);
        BETA_fit{ii} = stats.BETA(1:p,:)';
        D_fit{ii} = stats.BETA((p+1):end,:)';

        % (3) projection for constraint
        [Cout_trans,Dout_trans,~] = projectAB(C_fit{ii}',D_fit{ii}', q, 1);
        C_fit{ii} = Cout_trans';
        D_fit{ii} = Dout_trans';

        % (4) M-step: update c0, Q0, mc, Ac and Qc

        mu11 = zeros(q,q,T); % E(c_{t}*c'_{t})
        mu10 = zeros(q,q,T-1); % E(c_{t}c'_{t-1})
        for tt=1:T
            mu11(:,:,tt) =  W(:,:,tt) + C_fit{ii}(:,tt)*(C_fit{ii}(:,tt))';
            if tt>=2
                mu10(:,:,tt-1) = W01(:,:,tt-1) + C_fit{ii}(:,tt)*(C_fit{ii}(:,tt-1))';
            end
        end

        % 4a. c0 & Q0
        c0 = C_fit{ii}(:,1);
        Q0 = W(:,:,1);

        % 4b. mc & Ac
        smu1 = sum(C_fit{ii}(:,2:T),2);
        smu10 = sum(mu10,3);
        smu0 = sum(C_fit{ii}(:,1:(T-1)),2);
        smu00 = sum(mu11(:,:,1:(T-1)), 3);
        smu11 = sum(mu11(:,:,2:T), 3);

        mA_tmp = [smu1 smu10]/([T-1 smu0';smu0 smu00]);
%         mc = mA_tmp(:,1);
%         Ac = mA_tmp(:,2:end);

        % 4c. Qc
        Qc = (smu11 - mc*smu1' - Ac*smu10' - smu1*mc' - smu10*Ac' +...
            mc*mc' + Ac*smu0*mc' + mc*smu0'*Ac' + Ac*smu00*Ac')/(T-1);


        % (5) evaluate the llhd/spk
        MU_tmp = exp(X'*BETA_fit{ii}' + C_fit{ii}'*D_fit{ii}');
        LLHD(ii) = nansum(-MU_tmp + Y'.*log(MU_tmp + (MU_tmp == 0)), 'all')/ nansum(Y','all');

        change = abs(LLHD(ii-1)/LLHD(ii) - 1);
        if verbose; disp("Iter" + ii + ": " + LLHD(ii) + ", Qc(1) = " +...
                Qc(1,1) + ", llhd_change: " + change); end

        if ii > 10
            change_pre = abs(LLHD(ii-2)/LLHD(ii-1) - 1);
            if change > change_pre || change < tol
                break;
            end
        end


        %     if(abs(LLHD(ii-1)/LLHD(ii) - 1) < tol) && ii > 10
        %         break;
        %     end
    catch
        break
    end
end

% C. linear dynamics for X
% p = size(X,1);
% mdl = varm(p,1);
% res = estimate(mdl, X');
% mx = res.Constant;
% Ax = res.AR{1};
% Qx = res.Covariance;

output.c0 = c0;
output.Q0 = Q0;

output.BETA = BETA_fit{ii};
output.C = C_fit{ii};
output.D = D_fit{ii};
output.LLHD = LLHD(ii);
% output.mx = mx;
% output.Ax = Ax;
% output.Qx = Qx;

output.mc = mc;
output.Ac = Ac;
output.Qc = Qc;


end