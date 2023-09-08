function pCover = coverRate(theta_sorted, HDR)

nTheta = length(theta_sorted);
nCover = 0;

for bb = 1:nTheta
    nSeg = size(HDR{bb},1);
    for mm = 1:nSeg
        if (HDR{bb}(mm,1) <= theta_sorted(bb)) && (HDR{bb}(mm,2) >= theta_sorted(bb))
            nCover = nCover + 1;
        end
    end
end

pCover = nCover/nTheta;


end