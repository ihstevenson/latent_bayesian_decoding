% Evaluate coverage given posteriors and true 1d external variable
% Inputs:
%  ppp [x_size x trials] matrix of posterior probabilities
%  xGrid - vector for decoding grid
%  x_decode - vector of true values
%  credMass_vec (optional) - vector of nominated probabilities to calculate coverage for
%
% Output:
%   struct with results

function coverage_res = eval_coverage(ppp,xGrid,x_decode,credMass_vec)

if nargin<4
    coverage_res.credMass_vec = linspace(0.01,0.99,100);
else
    coverage_res.credMass_vec = credMass_vec;
end

for j=1:size(ppp,2)
    for i=1:length(coverage_res.credMass_vec)
        [coverage_res.HDR{i}{j}, coverage_res.expectedp(i,j)] = discreteHDR(xGrid, ppp(:,j)', coverage_res.credMass_vec(i));
    end
end

for i=1:length(coverage_res.credMass_vec)
    coverage_res.pCover(i) = coverRate(x_decode, coverage_res.HDR{i});
    % correction for mismatch between nominated probability and the actual probability of the credible sets
    coverage_res.pCover_corr(i) = coverage_res.pCover(i).*coverage_res.credMass_vec(i)./nanmean(coverage_res.expectedp(i,:));
end