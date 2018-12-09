function [X_tr_transformed,X_ts_transformed,Y_tr,Y_ts]=applyDKDA(X,X2,Y,trIdx,tsIdx,options,options2)
%% options.t=thr1;  options2.t=thr2;
%% trIdx and tsIdx are the index of training and testing in X and X2

%% apply KDA on the first feature type matrix (X) with thr = options.t 

fea=X(trIdx,:);gnd=Y(trIdx);
options.KernelType = 'Gaussian';
[eigvector, ~] = KDA(options,gnd,fea);
%         fea2=unique(X(trIdx,:),'rows');
feaTest = X(tsIdx,:);
Ktest = constructKernel(feaTest,X(trIdx,:),options);
X_ts= Ktest*eigvector;
Ktrain = constructKernel(fea,fea,options);
X_tr= Ktrain*eigvector;
%% apply KDA on the second feature type matrix (X2) with thr = options2.t 
fea=X2(trIdx,:);
options.KernelType = 'Gaussian';
[eigvector, ~] = KDA(options2,gnd,fea);

feaTest = X2(tsIdx,:);
Ktest = constructKernel(feaTest,fea,options2);
X_ts2= Ktest*eigvector;
Ktrain = constructKernel(fea,fea,options2);
X_tr2= Ktrain*eigvector;
%% generate transformed training/testing features with corresponding labels
X_tr_transformed=[X_tr X_tr2];
X_ts_transformed=[X_ts X_ts2];
Y_tr=Y(trIdx);Y_ts=Y(tsIdx);
