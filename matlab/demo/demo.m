% demo code 
% 
% (C) Martin Slawski, Nov 2013

%
addpath(genpath('../code'))

%%% TOY EXAMPLE %%%
% load randomly generated sparse psd M matrix

load('toy.mat')
%spy(Omegastar)
p = size(Omegastar, 2);

% generate random sample
n = 200;
X= mvnrnd(zeros(n, p), inv(Omegastar));
S = X' * X/n;

[Omega, Sigma, conv] = blockdescent_omega(S);
csvwrite('test.csv', Sigma);
% % plot objective and kkt optimality
% plot(1:length(conv.objs), log10(conv.objs), '-*')
% plot(1:length(conv.objs), log10(conv.kkts), '-*')
% % extract off-diagonal entries
% off = abs(extractoff(Omega));
% hist(off(off > 1E-3), 20)
% 
% % edgeset and its complement
% 
% Estar = find(extractoff(Omegastar) < 0);
% Estar_c = find(extractoff(Omegastar) > -eps);
% 
% % plot off-diagonal entries separately for edges/non-edges
% figure
% hold on
% plot(1:length(off(Estar_c)), off(Estar_c), 'x', 'color', 'red') % not corresponding to edges
% plot(linspace(1, length(off(Estar_c)), length(off(Estar))),  off(Estar), '*') % corresponding to edges
% 
% %************************************************%
% 
% %%% MAMMALS dataset %%%
% % dataset consists of n = 85 quantitative features (s. 'featurenames') 
% % (anatomical, ecological, ...)  of p = 50 mammals
% % (s. 'names')
% 
% load('animals.mat')
% % compute correlation matrix
% S = cov(X, 1);
% C = diag(1./sqrt(diag(S)))  * S * diag(1./sqrt(diag(S))) ;
% %
% [Omega, Sigma, conv] = blockdescent_omega(C);
% 
% % result is quite sparse 
% 
% imagesc(Omega)
% colorbar
% 
% % get pairs of mammals with high partial covariances
% 
% lowertriangle = tril(Omega);
% [negpcovs, ixso] = sort(lowertriangle(:), 'ascend');
% 
% [I20, J20] = ind2sub([50 50], ixso(1:20));
% 
% for i=1:20
%     
%     disp([names{I20(i)}  '--'  names{J20(i)}   ',  partial covariance: ' num2str(-negpcovs(i))])
%     
% end
% 
% %************************************************%
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 






% APPENDIX %
% generate sparse random psd M matrix
% p = 2^8;
% df =  p * (p - 1)/2;
% deltafac = 1.05;
% dens = 0.01;
% sparsevec = sprand(df, 1, dens);
% sparsevec = 1 * (sparsevec > eps);
% Mask = triu(~eye(p), 1);
% B = zeros(p);
% B(find(Mask)) = sparsevec;
% B = B + transpose(B);
% rB = max(eig(B));
% M = deltafac * rB * eye(p) - B; % M-matrix;
% Minv = M \ eye(p); % inverse of M.
% Dhalf = diag(sqrt( diag(Minv)) );
% Dinvhalf = diag(1 ./ diag(Dhalf));
% Sigmastar = Dinvhalf * Minv * Dinvhalf;
% Omegastar =  Dhalf * M * Dhalf;