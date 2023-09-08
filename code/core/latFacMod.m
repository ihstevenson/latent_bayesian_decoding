function output = latFacMod_ihs(Y,X,k,distribution,method, ProjVer, verbose, varargin)

% to debug
% Y = Y';
% X = X';
% k = 2;
% distribution = "NB";
% method = "minFunc";
% ProjVer = 1;
% verbose = true;

p = size(X,2);
t = size(Y,1);
n = size(Y,2);
reg = 1e-2;

if method ~= "minFunc" && method ~= "coorDesc"
    error("method should be 'minFunc' or 'coorDesc'");
end

tol = 1e-6;
maxIter = 1000;
if (~isempty(varargin))
    c = 1 ;
    while c <= length(varargin)
        switch varargin{c}
            case {'tol'}
                tol = varargin{c+1};
            case {'maxIter'}
                maxIter = varargin{c+1};
            case {'reg'}
                reg = varargin{c+1};
        end % switch
        c = c + 2;
    end % for
end % if

if k==0
    output_tmp = glmMod(Y,X,distribution);
    output.BETA = output_tmp.BETA;
    if distribution == "NB"
        output.alpha = output_tmp.ALPHA(1,:)';
    end
    return;
end

% initialization
A_fit{1} = zeros(t,k); % avoid overfitting of AB, set them = 0 initially
B_fit{1} = zeros(k,n);
dTrace = zeros(k, maxIter);
BETA_fit{1} = zeros(p,n);
LLHD = zeros(maxIter, 1);

if distribution == "NB"
    alpha_fit = zeros(n,maxIter);
    alpha_fit(:,1) = ones(n,1)*0.01;

    MU = @(BETA, A,B) exp(X*BETA + A*B);
    ALPHA = @(alpha) repmat(alpha', t, 1);
    MU_tmp = MU(BETA_fit{1}, A_fit{1}, B_fit{1});
    ALPH_tmp = ALPHA(alpha_fit(:,1));
    AMU_tmp = ALPH_tmp.*MU_tmp;
    AMUp1_tmp = 1 + AMU_tmp;

    LLHD(1) = nansum(Y.*log(AMU_tmp./AMUp1_tmp) - (1./ALPH_tmp).*log(AMUp1_tmp) +...
        gammaln(Y + 1./ALPH_tmp) - gammaln(Y + 1) - gammaln(1./ALPH_tmp), 'all')/ nansum(Y,'all');

elseif distribution == "Poisson"
    LAM = @(BETA, A,B) exp(X*BETA + A*B);
    LAM_tmp = LAM(BETA_fit{1}, A_fit{1}, B_fit{1});
    LLHD(1) = nansum(-LAM_tmp + Y.*log(LAM_tmp + (LAM_tmp == 0)), 'all')/ nansum(Y,'all');
else
    error('distribution should be "NB" or "Poisson"');
end

if method == "minFunc"
    options.maxIter = 50;
    options.MaxFunEvals = 100;
    options.Display = 'off';
end

for ii = 2:maxIter

%     try
        % (1) covariates
        BETA_fit{ii} = BETA_fit{ii-1};
        OFFSET = A_fit{ii-1}*B_fit{ii-1};
        for nn = 1:n
            idxTmp = ~isnan(Y(:,nn));
            if distribution == "NB"
                stats = nbreg(X(idxTmp,:), Y(idxTmp,nn), 'offset', OFFSET(idxTmp,nn),...
                    'regularization', reg);
                BETA_fit{ii}(:,nn) = stats.b;
                alpha_fit(nn,ii) = stats.alpha;
            else
                %                 BETA_fit{ii}(:,nn) = glmfit(X(idxTmp,:),Y(idxTmp,nn),...
                %                     'poisson','link','log','Constant','off', 'offset', OFFSET(idxTmp,nn));
                stats = nbreg(X(idxTmp,:), Y(idxTmp,nn), 'offset', OFFSET(idxTmp,nn),...
                    'distr', 'poisson', 'regularization', reg);
                BETA_fit{ii}(:,nn) = stats.b;

            end
        end

        % (2) latent factor: unconstraint + projection
        OFFSET = X*BETA_fit{ii};

        if distribution == "NB"
            [x,~,~,~] = minFunc(@lossLowRank_NB,[A_fit{ii-1}(:); B_fit{ii-1}(:)],...
                options,Y,k,alpha_fit(:,ii),OFFSET,0);
        else
            [x,~,~,~] = minFunc(@lossLowRank_poi,[A_fit{ii-1}(:); B_fit{ii-1}(:)],...
                options,Y,k,OFFSET);
        end
        A_tmp = reshape(x(1:(t*k)),t,k);
        B_tmp = reshape(x((t*k+1):end),k,n);
        [A_fit{ii},B_fit{ii}, dTrace(:,ii)] = projectAB(A_tmp,B_tmp, k, ProjVer);


        % (3) check convergence
        if distribution == "NB"
            MU_tmp = MU(BETA_fit{ii}, A_fit{ii}, B_fit{ii});
            ALPH_tmp = ALPHA(alpha_fit(:,ii));
            AMU_tmp = ALPH_tmp.*MU_tmp;
            AMUp1_tmp = 1 + AMU_tmp;
            LLHD(ii) = nansum(Y.*log(AMU_tmp./AMUp1_tmp) - (1./ALPH_tmp).*log(AMUp1_tmp) +...
                gammaln(Y + 1./ALPH_tmp) - gammaln(Y + 1) - gammaln(1./ALPH_tmp), 'all')/ nansum(Y,'all');
        else
            LAM_tmp = LAM(BETA_fit{ii}, A_fit{ii}, B_fit{ii});
            LLHD(ii) = nansum(-LAM_tmp + Y.*log(LAM_tmp + (LAM_tmp == 0)), 'all')/ nansum(Y,'all');
        end

        if verbose; disp("Iter" + ii + ": " + LLHD(ii)); end

        if(abs(LLHD(ii-1)/LLHD(ii) - 1) < tol)
            if method == "minFunc"
                method = "coorDesc";
                continue;
            elseif ii > 10
                break;
            end
        end

%     catch
%         ii = ii-1;
%         break;
%     end
end

output.BETA = BETA_fit{ii};
output.A = A_fit{ii};
output.B = B_fit{ii};
output.d = dTrace(:,ii);
output.LLHD = LLHD(ii);

output.BETA_trace = BETA_fit;
output.A_trace = A_fit;
output.B_trace = B_fit;
output.d_trace = dTrace(:,1:ii);
output.LLHD_trace = LLHD(1:ii);

if distribution == "NB"
    output.alpha = alpha_fit(:,ii);
    output.alpha_trace = alpha_fit(:,1:ii);
end


end