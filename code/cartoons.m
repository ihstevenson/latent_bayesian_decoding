
%% Fig 2 - Illustration of single neuron posteriors influenced by latent variable

xgrid=linspace(-pi,pi,256);
tf=@(x)8*exp(cos(x))+10;

n=1000;
xsamp = rand(n,1)*2*pi-pi;
zsamp = rand(n,1)*10-5;
ysamp=poissrnd(tf(xsamp)+zsamp);
[~,~,i]=unique(zsamp);

figure(1)
clf
scatter(xsamp*180/pi,ysamp,20,i,'filled','MarkerFaceAlpha',0.5)
hold on
plot(xgrid*180/pi,tf(xgrid),'LineWidth',2)
box off; set(gca,'TickDir','out')
xlim([-1 1]*180)
colormap turbo

figure(2)
% tuning curves
subplot(3,6,[1 2 7 8 13 14])
zvec=[-0.2 -0.1 0 0.1 0.2];
cmap = parula(length(zvec));
for i=1:length(zvec)
    plot(xgrid*180/pi,tf(xgrid)*exp(zvec(i)),'LineWidth',2,'Color',cmap(i,:))
    hold on
    if zvec(i)==0
        plot(xgrid*180/pi,tf(xgrid)*exp(zvec(i)),'k--','LineWidth',2)
    end
end
hold off
box off; set(gca,'TickDir','out')
xlim([-1 1]*180)
set(gca,'XTick',[-180:90:180]);

% posteriors with specific z
yobsvec=[35:-10:15];
for j=1:3
    subplot(3,6,(j-1)*6+3)
    yobs=yobsvec(j);
    for i=1:length(zvec)
        lam = tf(xgrid)*exp(zvec(i));
        logp=(yobs*log(lam)-lam);
        plot(xgrid*180/pi,exp(logp)/sum(exp(logp)),'LineWidth',2,'Color',cmap(i,:))
        hold on
        if zvec(i)==0
            plot(xgrid*180/pi,exp(logp)/sum(exp(logp)),'k--','LineWidth',2)
        end
    end
    hold off
    box off; set(gca,'TickDir','out')
    xlim([-1 1]*180)
    set(gca,'XTick',[-180:90:180]);
    ylim([0 0.028])


    zgrid=linspace(-1,1,256);
    sig=0.25;
    [X,Z]=meshgrid(xgrid,zgrid);
    lam = tf(X).*exp(Z);
    logpxz = (yobs*log(lam)-lam+gammaln(yobs+1)-Z.^2/2/sig.^2-log(sig*sqrt(2*pi)));
    pxz = exp(logpxz)./sum(exp(logpxz(:)));

    % figure(3)
    % joint p(y|theta,z)
    subplot(3,6,(j-1)*6+4)
    imagesc(xgrid*180/pi,zgrid,pxz)
    set(gca,'YDir','normal')
    box off; set(gca,'TickDir','out')
    set(gca,'XTick',[-180:90:180]);

    % marginalize over theta
    subplot(3,6,(j-1)*6+5)
    plot(zgrid,sum(pxz,2),'LineWidth',2)
    hold on
    axis tight
    line([0 0],ylim(),'Color','k','LineWidth',2)
    hold off
    box off; set(gca,'TickDir','out')
    ylim([0 0.02])

    % marginalize over z
    subplot(3,6,(j-1)*6+6)
    plot(xgrid*180/pi,sum(pxz),'LineWidth',2)
    box off; set(gca,'TickDir','out')
    xlim([-1 1]*180)
    hold on
    lam = tf(xgrid)*exp(0);
    logp=(yobs*log(lam)-lam);
    plot(xgrid*180/pi,exp(logp)/sum(exp(logp)),'k','LineWidth',2)
    hold off
    set(gca,'XTick',[-180:90:180]);
    ylim([0 0.02])
end

% exportgraphics(gcf,'latent_decoding_cartoon.pdf','ContentType','vector')

%% Fig 1A - sketch of coverage

rng(3)
HDR = cell(5,3);
xgrid=linspace(-10,10,256);
figure(4)
clf
cmap = lines(8);
for i=1:5
%     subplot(5,1,i)
    r = rand(1)*9-5;
    s = chi2rnd(10,1)/10;
    p1 = normpdf(xgrid,r,sqrt(4*s));
    plot(xgrid/max(xgrid),p1+2*i,'LineWidth',2,'Color',cmap(1,:))
    HDR{i,1} = discreteHDR(xgrid, p1/sum(p1), 0.95);
    hold on

    s = chi2rnd(10,1)/10;
    p2 = normpdf(xgrid,r,sqrt(1*s));
    plot(xgrid/max(xgrid),p2+2*i,'LineWidth',2,'Color',cmap(3,:))
    HDR{i,2} = discreteHDR(xgrid, p2/sum(p2), 0.95);
    
    
    s = chi2rnd(10,1)/10;
    p2 = normpdf(xgrid,r,sqrt(0.2*s));
    plot(xgrid/max(xgrid),p2+2*i,'LineWidth',2,'Color',cmap(2,:))
    HDR{i,3} = discreteHDR(xgrid, p2/sum(p2), 0.95);
    

    box off; set(gca,'TickDir','out')
    xlim([-1 1]*0.75)

    line(HDR{i,1}/max(xgrid),HDR{i,1}*0+1+2*i,'LineWidth',2,'Color',cmap(1,:))
    line(HDR{i,2}/max(xgrid),HDR{i,2}*0+1.1+2*i,'LineWidth',2,'Color',cmap(3,:))
    line(HDR{i,3}/max(xgrid),HDR{i,3}*0+1.2+2*i,'LineWidth',2,'Color',cmap(2,:))
end
hold off
line([0 0],ylim())

% exportgraphics(gcf,'coverage_cartoon.pdf','ContentType','vector')


%% Fig 1B - coverage curve illustration

ci_vec = linspace(0.01,0.99,100);
inc = zeros(1000,100,3);
for i=1:size(inc,1)
    r = randn(1);
    s = chi2rnd(10,1)/10;
    p1 = normpdf(xgrid,r,sqrt(4*s));

    s = chi2rnd(10,1)/10;
    p2 = normpdf(xgrid,r,sqrt(1*s));

    s = chi2rnd(10,1)/10;
    p3 = normpdf(xgrid,r,sqrt(0.2*s));

    for j=1:length(ci_vec)
        hdr = discreteHDR(xgrid, p1/sum(p1), ci_vec(j));
        inc(i,j,1) = hdr(1)<0 & hdr(2)>0;
        hdr = discreteHDR(xgrid, p2/sum(p2), ci_vec(j));
        inc(i,j,2) = hdr(1)<0 & hdr(2)>0;
        hdr = discreteHDR(xgrid, p3/sum(p3), ci_vec(j));
        inc(i,j,3) = hdr(1)<0 & hdr(2)>0;
    end
end

figure(6)
plot(ci_vec,squeeze(mean(inc)))
line([0 1],[0 1])
box off; set(gca,'TickDir','out')
% exportgraphics(gcf,'coverage_curve.pdf','ContentType','vector')