function logMar = logMarA_poi_v2(offset_t, yt, B_mle)

% to debug
% offset_t = (xToZ(xGrid(tt))*BETA_lat)';
% yt = y_star;
% B_mle = B_lat;

offset_t = offset_t(:);
stats = nbreg(B_mle',yt,'offset',offset_t,...
            'distr', 'poisson', 'reg', 10e-1);
a_mle = stats.b;

lamt = exp(offset_t + B_mle'*a_mle);
Sig = inv(B_mle*diag(lamt)*B_mle');
logMar = sum(-lamt + yt.*log(lamt + (lamt == 0))) + 0.5*log(det(Sig));

end