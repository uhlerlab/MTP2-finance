% Code for solving the optimization problem
% 
% min_{\Omega}  f(\Omega) sb. to off(\Omega) <= 0
% 
% where
% 
% f(\Omega) = -log det \Omega + tr(\Omega S)
% 
% is the log determinant divergence, the
% Bregman divergence on the positive semidefinite
% cone induced by the log determinant.
%
% Algorithm is based on cyclical block coordinate descent,
% with one column/row being optimized at a time.
% A detailed description is given in  
%
% 'Positive definite M-matrices and structure learning in attractive
% Gaussian Markov random fields'
% 
%
% Input: 
% 
% S:     covariance matrix of the sample
%
% nnlstol: tolerance (KKT optimality) used for each block update. Default:
%
% stoptol: Stop if 
%              
%              f(\Omega^{k+1}) - f(\Omega^k) < stopol. Default: 
%
% where  f(\Omega^r) is the iterate after completing
%  the r-th cycle.
%
% 
% Output:
%
% Omega     approximate minimizer found by the algorithm 
% Sigma      the inverse of Omega
% 
% conv        a struct having the fields 'kkts', 'objs', 'tols'. 
%                
%                'kkts': KKT optimality of the sequence of iterates (after each completed cycle)
%                'objs': objective values of the sequence of iterates (after each completed cycle)
%                'tols': ||\Omega^{k+1}  - \Omega^k||, k=1,2,... where || . || is the spectral norm.
%
%
%  (C) Martin Slawski, Nov 2013

function [Omega, Sigma, conv] = blockdescent_omega(S, nnlstol, stoptol)

%%% set default parameters

if nargin < 3
    stoptol = 1E-4;
end

if nargin < 2
    nnlstol = 1E-6;
end

%%%
p = size(S, 2);

%%% starting value;
Omega = diag(1 ./ diag(S));     % S \ eye(p);
Sigma = diag(diag(S));  

%%%
obj = @(Z) -log_det(Z) + trace(Z * S);
%%%
objs = [];
tols = [];
kkts = [];
%%%
converged = false;
%
% set options for NNLS solver
o_g = initopt_general('mode', 1, 'tol', nnlstol);
o_s = opt_blockpivoting();
%
cycle = 0;
%
while ~converged
    
    Omegaold = Omega;

    for j=1:p
        
        %%%
        minusj = setdiff(1:p,j);
        
        s_jj = S(j,j);
        s_j = S(minusj,j);
        
        sigma_j = Sigma(minusj, j);
        sigma_jj = Sigma(j,j);
        G = Sigma(minusj, minusj) - (sigma_j * sigma_j'./ sigma_jj);
        
        %%% quantities to be passed into an NNLS solver.
        AtA = s_jj * G; 
        Atb = s_j; 
        %%% use block-pivoting
        out = GGM_blockpivoting_down(AtA, Atb, o_g, o_s);
        omega_j = -out.xopt; %%% sign flip
        qpart = G * omega_j;
        q = omega_j' * qpart;
        omega_jj = (1 + s_jj * q)/s_jj; %%% resolve
        %%% Update the current Omega
        Omega(j,j) = omega_jj;
        Omega(minusj, j) = omega_j;
        Omega(j, minusj) = omega_j;
        %%% Update the current Sigma
        Sigma(j,j) = s_jj;
        Sigma(j, minusj) = -s_jj * qpart; 
        Sigma(minusj, j) = Sigma(j, minusj);
        %%% use Sherman-Woodbury
        Sigma(minusj, minusj) = (G + (qpart * qpart' / (omega_jj - q)));
        %
        
    end
    
    
    %%% kkt optimality
    
    M = Omega < -nnlstol;
    Mtilde = 1 - M; 
    
    kkt1 = max(max(abs(Sigma - S) .* M));
    kkt2 = max(max(max(abs(min((Sigma  - S) .* Mtilde, 0)))));
    
    kkt = max(kkt1, kkt2);
    
    kkts = [kkts kkt];
        
    %%%
    
    objs = [objs obj(Omega)]; 
    cycle = cycle + 1;
    
    disp(['Cycle: ' num2str(cycle) ' Objective: ' num2str(objs(end))])
    
    tols = [tols norm(Omegaold - Omega)]; %%% check operator norm.
    
    if length(objs) > 1
        if(objs(end-1) - objs(end) < stoptol)
            converged = true;
        end    
    end

end

conv.objs = objs;
conv.tols = tols;
conv.kkts = kkts;

