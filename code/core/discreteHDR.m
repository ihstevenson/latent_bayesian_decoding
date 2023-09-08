function [HDR,expectedp] = discreteHDR(xGrid, pos_norm, credMass)

pos_norm_sort = sort(pos_norm,'descend');
pos_cut = pos_norm_sort(find(cumsum(pos_norm_sort) >= credMass, 1));

if ~isempty(pos_cut)
    intervals = find(pos_norm>=pos_cut);
    expectedp = sum(pos_norm(pos_norm>=pos_cut));
    HDR(:,1) = xGrid(intervals([true diff(intervals)>1]));
    HDR(:,2) = xGrid(intervals([diff(intervals)>1 true]));
else
    HDR=[];
    expectedp = 0;
end

end