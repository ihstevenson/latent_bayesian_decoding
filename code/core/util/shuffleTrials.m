
function yshuff = shuffTrials(y,x)

yshuff=zeros(size(y));
[ux,ia,ic] = unique(x);

for i=1:length(ux)
    for j=1:size(y,2)
        idx = ic==i;
        ridx = randperm(length(idx));
        yshuff(idx,j) = y(idx(ridx),j);
    end
end