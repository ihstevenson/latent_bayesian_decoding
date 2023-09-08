
function plotSegmentErrorBar(x,y,nseg,addeb)

px = prctile(x,linspace(0,1,nseg+1)*100);
[N,edges,bin] = histcounts(x,px);


for i=1:nseg
    mx(i)= mean(x(bin==i));
    my(i)=mean(y(bin==i));
    ey(i)=std(y(bin==i));
end
% errorbar(mx,my,ey,'CapSize',0)
plot(mx,my,'-')
if addeb
    hold on
    plot(mx,my+ey,'k:')
    plot(mx,my-ey,'k:')
end