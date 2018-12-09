function [DS2_tr,DS2_ts, discrimVec, eigValues, max2] = lda6(DS_tr,DS_ts, discrimVecNum)% adjusted from lda3 for RT scenario
%lda: Linear discriminant analysis
%
%	Usage:
%		DS2 = lda(DS)
%		DS2 = lda(DS, discrimVecNum)
%		[DS2, discrimVec, eigValues] = lda(...)
%
%	Description:
%		DS2 = lda(DS, discrimVecNum) returns the results of LDA (linear discriminant analysis) on DS
%			DS: input dataset (Try "DS=prData('iris')" to get an example of DS.)
%			discrimVecNum: No. of discriminant vectors
%			DS2: output data set, with new feature vectors
%		[DS2, discrimVec, eigValues] = lda(DS, discrimVecNum) returns extra info:
%			discrimVec: discriminant vectors identified by LDA
%			eigValues: eigen values corresponding to the discriminant vectors
%
%	Example:
%		% === Scatter plots of the LDA projection
% 		DS=prData('wine');
% 		DS2=lda(DS);
% 		DS12=DS2; DS12.input=DS12.input(1:2, :);
% 		subplot(1,2,1); dsScatterPlot(DS12); xlabel('Input 1'); ylabel('Input 2');
% 		title('Wine dataset projected on the first 2 LDA vectors'); 
% 		DS34=DS2; DS34.input=DS34.input(end-1:end, :);
% 		subplot(1,2,2); dsScatterPlot(DS34); xlabel('Input 3'); ylabel('Input 4');
% 		title('Wine dataset projected on the last 2 LDA vectors');
% 		% === Leave-one-out accuracy of the projected dataset using KNNC
% 		fprintf('LOO accuracy of KNNC over the original wine dataset = %g%%\n', 100*perfLoo(DS, 'knnc'));
% 		fprintf('LOO accuracy of KNNC over the wine dataset projected onto the first two LDA vectors = %g%%\n', 100*perfLoo(DS12, 'knnc'));
% 		fprintf('LOO accuracy of KNNC over the wine dataset projected onto the last two LDA vectors = %g%%\n', 100*perfLoo(DS34, 'knnc'));
%
%	Reference:
%		[1] J. Duchene and S. Leclercq, "An Optimal Transformation for Discriminant Principal Component Analysis," IEEE Trans. on Pattern Analysis and Machine Intelligence, Vol. 10, No 6, November 1988
%
%	See also ldaPerfViaKnncLoo.

%	Category: Data dimension reduction
%	Roger Jang, 19990829, 20030607, 20100212

if nargin<1, selfdemo; return; end
if ~isstruct(DS_tr)
	fprintf('Please try "DS=prData(''iris'')" to get an example of DS.\n');
	error('The input DS should be a structure variable!');
end
if ~isstruct(DS_ts)
	fprintf('Please try "DS=prData(''iris'')" to get an example of DS.\n');
	error('The input DS should be a structure variable!');
end
if nargin<3 &&size(DS_tr.input,1)>400
    discrimVecNum=400;
elseif nargin<3&&size(DS_tr.input,1)<=400
    discrimVecNum=size(DS_tr.input,1);
elseif discrimVecNum<1
  discrimVecNum=round(size(DS_tr.input,1)*discrimVecNum);  
end

% ====== Initialization
m = size(DS_tr.input,1);	% Dimension of data point
n = size(DS_tr.input,2);	% No. of data point
A = DS_tr.input;
A_ts=DS_ts.input;
if size(DS_tr.output, 1)==1	% Crisp output
	classLabel = DS_tr.output;
	[diffClassLabel, classSize] = elementCount(classLabel);
	classNum = length(diffClassLabel);
	mu_tr = mean(A, 2);
    mu_ts=mean(A_ts,2);

	% ====== Compute B and W
	% ====== B: between-class scatter matrix
	% ====== W:  within-class scatter matrix
	% M = \sum_k m_k*mu_k*mu_k^T
	M = zeros(m, m);
	for i = 1:classNum,
		index = find(classLabel==diffClassLabel(i));
		classMean = mean(A(:, index), 2);
		M = M + classSize(i)*(classMean*classMean');
	end
	W = A*A'-M;
	B = M-n*(mu_tr*mu_tr'); 
else % Potential fuzzy output
	% Put fuzzy lda code here
end

% ====== Find the best discriminant vectors
if det(W)==0
% 	error('W is singular. One possible reason: a feature has the same value across all training data.'); 
end
invW = inv(W);
Q = W\B;

D = [];
for i = 1:discrimVecNum
    if mod(i,50)==0
    disp(sprintf('in LDA:discrimVecNum %d of %d ',i,(discrimVecNum)));
    end
	[eigVec, eigVal] = pca(Q);
	[ee, index] = max(diag(eigVal)); 
    max2(i)=index;
    if isreal(ee)
        eigValues(i)=ee;
%         index
        D = [D, eigVec(:, index)];		% Each col of D is a eigenvector
        Q = (eye(m)-invW*D*inv(D'*invW*D)*D')*invW*B;
    else
        discrimVecNum=i-1;
%          Q = (eye(m)-invW*D*inv(D'*invW*D)*D')*invW*B;
        break;
    end
end
DS2_tr=DS_tr;
DS2_tr.input = D(:,1:discrimVecNum)'*A; 

DS2_ts=DS_ts;
DS2_ts.input = D(:,1:discrimVecNum)'*A_ts; 

discrimVec = D;
DS2_tr.name=strcat(DS2_tr.name,'_LDA_', num2str(discrimVecNum));
DS2_ts.name=strcat(DS2_ts.name,'_LDA_', num2str(discrimVecNum));
% [val,max2]=max(D);
% ====== Self demo
function selfdemo
mObj=mFileParse(which(mfilename));
strEval(mObj.example);
