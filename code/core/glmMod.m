function output = glmMod(Y,X,distribution, varargin)

% to debug
% Y = y_encode;
% X = z_encode;
% distribution = "NB"; % "Poisson" "NB"
% reg = 1e-4;

modelAlpha = false;
G = [];
reg = 1e-4;

if (~isempty(varargin))
    c = 1 ;
    while c <= length(varargin)
        switch varargin{c}
            case {'modelAlpha'}
                modelAlpha = varargin{c+1};
                if distribution == "Poisson" && modelAlpha
                    error('Poisson has no alpha');
                end
            case {'G'}
                G = varargin{c+1};
            case {'reg'}
                reg = varargin{c+1};
        end % switch
        c = c + 2;
    end % for
end % if

p = size(X,2);
t = size(Y,1);
n = size(Y,2);

if size(G,2)==1 && sum(G(:,1)==1) == t
    modelAlpha = false;
    G = [];
    disp("G is all 1, equivalent to modelAlpha = false");
end

BETA = zeros(p,n);
if distribution == "NB"
    ALPHA = zeros(t,n);
    if modelAlpha && ~isempty(G)
        q = size(G,2);
        GAM = zeros(q,n);
        maxIter = 1000;
        tol = 1e-8;
        options_gam.maxIter = 500;
        options_gam.MaxFunEvals = 1000;
        options_gam.Display = 'off';
    end
end

for ii = 1:n
    if distribution == "Poisson"
        
        stats = nbreg(X, Y(:,ii), 'distr', 'poisson', 'regularization', reg);
        BETA(:,ii) = stats.b;
        
    elseif distribution == "NB"
        
        if modelAlpha && ~isempty(G)
            
            % initialization...
            fit = nbreg(X, Y(:,ii));
            beta_tmp = fit.b;
            gam_tmp = [log(fit.alpha);zeros(q-1,1)];
            mu_tmp = exp(X*beta_tmp);
            alph_tmp = exp(G*gam_tmp);
            amu_tmp = alph_tmp.*mu_tmp;
            amup1_tmp = 1 + amu_tmp;
            llhd_tmp = nansum(Y(:,ii).*log(amu_tmp./amup1_tmp) - (1./alph_tmp).*log(amup1_tmp) +...
                gammaln(Y(:,ii) + 1./alph_tmp) - gammaln(Y(:,ii) + 1) - gammaln(1./alph_tmp), 'all');
            
            for gg = 2:maxIter
                
                % (1) fit gamma
                mu_tmp = exp(X*beta_tmp);
                try
                    options_gam.Method = 'newton';
                    [gam_tmp,~,~,~] = minFunc(@mleAlph,gam_tmp,...
                        options_gam,Y(:,ii),mu_tmp,G);
                catch
                    options_gam.Method = 'lbfgs';
                    [gam_tmp,~,~,~] = minFunc(@mleAlph,gam_tmp,...
                        options_gam,Y(:,ii),mu_tmp,G);
                end
                
                % (2) fit beta
                alph_tmp = exp(G*gam_tmp);
                fit_tmp = nbreg(X,Y(:,ii),'alpha', alph_tmp, 'estAlpha', false,...
                    'regularization', reg);
                beta_tmp = fit_tmp.b;
                
                % (3) check convergence
                mu_tmp = exp(X*beta_tmp);
                amu_tmp = alph_tmp.*mu_tmp;
                amup1_tmp = 1 + amu_tmp;
                llhd_tmp_c = nansum(Y(:,ii).*log(amu_tmp./amup1_tmp) - (1./alph_tmp).*log(amup1_tmp) +...
                    gammaln(Y(:,ii) + 1./alph_tmp) - gammaln(Y(:,ii) + 1) - gammaln(1./alph_tmp), 'all');
                
                if(abs(llhd_tmp/llhd_tmp_c - 1) < tol)
                    break;
                end
                llhd_tmp = llhd_tmp_c;
            end
            
            BETA(:,ii) = beta_tmp;
            GAM(:,ii) = gam_tmp;
            ALPHA(:,ii) = exp(G*gam_tmp);
            
        else
            
            stats = nbreg(X, Y(:,ii), 'regularization', reg);
            BETA(:,ii) = stats.b;
            ALPHA(:,ii) = stats.alpha*ones(t,1);
        end
    else
        error('distribution should be "NB" or "Poisson"');
    end
end

output.BETA = BETA;

if distribution == "NB"
    output.ALPHA = ALPHA;
    if modelAlpha && ~isempty(G)
        output.GAM = GAM;
    end
end



end