% Main wrapper for Bayesian decoding
% Inputs:
%  trial_x_full [trials x 1] vector for external variables
%  trial_y_full [trials x neurons] matrix of spike counts
%  xGrid - vector for decoding grid
%  xToZ - function to represent x as a basis
%  kfold_cv - number of folds for cross-validation
%  latent_dim - dimensionality to use for the latent variable models
%
% Output:
%   struct with results for each cross-validation fold
%       res.ols_res(cvk), res.lat_res(cvk)
%   and errors and posteriors aligned to trial_x_full   

function res = decode_cv_cat(trial_x_full,trial_y_full,xGrid,xToZ,kfold_cv,latent_dim,reg)

res.cv = cvpartition(size(trial_y_full,1),'KFold',kfold_cv);

if nargin<7,
    reg=10e-2;
end

for cvk=1:kfold_cv
    x_encode = trial_x_full(res.cv.training(cvk));
    x_decode = trial_x_full(res.cv.test(cvk));
    y_encode = trial_y_full(res.cv.training(cvk),:);
    y_decode = trial_y_full(res.cv.test(cvk),:);
    z_encode = xToZ(x_encode);

    res.ols_res(cvk) = decode_ols_cat(z_encode,y_encode,x_decode,y_decode,xGrid,xToZ,reg);
    res.lat_res(cvk) = decode_latent_cat(z_encode,y_encode,x_decode,y_decode,latent_dim,xGrid,xToZ,reg);

    
    res.ols_res_all.Poippp(:,res.cv.test(cvk))=res.ols_res(cvk).Poippp;
    res.ols_res_all.NBppp(:,res.cv.test(cvk))=res.ols_res(cvk).NBppp;

    res.lat_res_all.Poippp(:,res.cv.test(cvk))=res.lat_res(cvk).Poippp;
    res.lat_res_all.NBppp(:,res.cv.test(cvk))=res.lat_res(cvk).NBppp;
end