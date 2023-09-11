function [x,fx,dfdx,xx,eoi] = newtonGH(fdf,x0,TolX,MaxIter, verbose)

% to debug
% fdf = gradHess;
% x0 = dX_tmp(:);
% TolX = 1e-4;
% MaxIter = 2;
% 

TolFun=eps;
xx(:,1) = x0;
dh = feval(fdf, x0);
fx = dh{1};
% disp(norm(fx))
for k = 1:MaxIter
    if verbose
        disp("norm fx = " + norm(fx))
    end
    dfdx = dh{2};
    dx = -dfdx\fx;
    dx(~isfinite(dx))=0;
    xx(:,k+1) = xx(:,k)+dx;
%     dhPre = dh;
    dh = feval(fdf,xx(:,k+1));
    fx = dh{1};
    
    if(sum(isnan(xx(:,k+1)))>0)
        break;
    end
    
    if(norm(fx)<TolFun || norm(dx) < TolX)
        break;
    end
    
    
end
warning('on');

eoi = k;
x = xx(:,k + 1);
dfdx = dh{2};
if(k == MaxIter)
    fprintf('The best in %d iterations\n',MaxIter)
end

end