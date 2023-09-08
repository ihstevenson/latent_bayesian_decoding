function pall = normlogP(logp_tmp)

covFun=max(logp_tmp');
logzall=(covFun+log(sum(exp(logp_tmp'-covFun))));
pall = exp(logp_tmp'-logzall);


end