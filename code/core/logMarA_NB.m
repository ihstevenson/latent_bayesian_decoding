function logMar = logMarA_NB_v2(offset_t, yt, B_mle, alpha_mle)

% to debug
% offset_t = (xToZ(xGrid(tt))*NBMod{cc}.BETA)';
% yt = y_star;
% B_mle = NBMod{cc}.B;
% alpha_mle =NBMod{cc}.alpha;

offset_t = offset_t(:);
stats = nbreg(B_mle',yt,'offset',offset_t,...
            'alpha', alpha_mle, 'estAlpha', false,'reg', 10e-1);
a_mle = stats.b;

mu_t = exp(offset_t + B_mle'*a_mle);
amu = alpha_mle.*mu_t;
amup1 = 1 + amu;
llhd = nansum(yt.*log(amu./amup1) - (1./alpha_mle).*log(amup1) +...
    gammaln(yt + 1./alpha_mle) - gammaln(yt + 1) - gammaln(1./alpha_mle));
Sig = inv(B_mle*diag(mu_t./amup1)*B_mle');
logMar = llhd + 0.5*log(det(Sig));

end