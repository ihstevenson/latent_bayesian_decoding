
function coverage_res = coverage_r(ppp,xGrid,x_decode,credMass_vec,doCorrection)

if nargin<5,
    doCorrection=false;
end

if nargin<4
    credMass_vec = linspace(0.01,0.99,100);
else
    credMass_vec = credMass_vec;
end

for j=1:size(ppp,2)
    for i=1:length(credMass_vec)
        [HDR{i}{j},expectedp(i,j)] = discreteHDR(xGrid, ppp(:,j)', credMass_vec(i));
    end
end

for i=1:length(credMass_vec)
    coverage_res(i) = coverRate(x_decode, HDR{i});
    if doCorrection
        coverage_res(i) = coverage_res(i)*credMass_vec(i)./nanmean(expectedp(i,:));
    end
end