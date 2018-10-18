% Block pivoting method (Portugal et al., 1994)
%
% - Method only works when the full Gram matrix can be kept in storage,
% - Method is not guaranteed to converge if the matrix A does not have full
%   rank, as pointed out in the paper and confirmed by experiments.

% Replace all index of opeations by binary operations.
 
function[out] = GGM_blockpivoting_down(A, b, options_general, options_specific)

if options_general.mode
   AtA = A;
   Atb = b;
end

%%% Initialize CPU time
t0 = tic;
%%%

if  options_specific.qbar <= 0
    qbar = 10;
else
    qbar =  options_specific.qbar;
end

% initialization
[n, p] = size(A);

if ~options_general.mode
    %%% un-pack Gram matrix if available; otherwise, compute Gram matrix. 
   if ~isempty(options_general.gram)
            AtA = options_general.gram;
        if ~all(size(AtA) == [p p])
            error('wrong input for the Gram matrix');
        end
        options_general.gram = [];
       else
        AtA = A'*A;
    end
end
%%%

%initialize variables
if ~options_general.mode
    Atb = A'*b;
end

global f;
if options_general.mode 
    f = @(x) x' * (AtA * x - 2 * Atb);  
else
    f = @(x) norm(A * x - b)^2;
end

%%% 
if options_general.mv == 0
   if 1.1 * 2 * n < p
      mv = 2; 
   else
      mv = 1; 
   end
else
    mv = options_general.mv;   
end

maxiter = options_specific.itmax;
F = false(p, 1);
Lambda = true(p, 1);
lambda = -Atb;
x = zeros(p, 1);
q = qbar;
ninf = p + 1;
iter = 0;


flag = 0; 

% gradient = lambda
%perf = getL(options_general.perfmeas, lambda, x);
%perf = perf(~isnan(perf));
%ConvSpeed =  [0 perf];
% fprev = inf; xprev = x + inf;
check = getkktopt(lambda, x);
%check_termination(t0, options_general, perf, lambda, x, inf, x + inf);
% check 
if check < options_general.tol
   out.xopt = x; 
   %out.err = f(x);
   %out.ConvSpeed = ConvSpeed; 
   out.check = check;
end

while (check > options_general.tol) && iter < maxiter
    fH1 = x < -eps;
    H1 = fH1 & F;
    
    
    fH2 = lambda < -eps;
    H2 = fH2 & Lambda; 
    H1H2 = H1 | H2;
    cardH1H2 = sum(H1H2);
    if cardH1H2 < ninf
        ninf = cardH1H2;
        q = qbar;
    else
        if q > 0
           q = q - 1;
        else
            if sum(F) >= min(n,p)
                r = find(H1, 1, 'last');
            else
                r = find(H1H2, 1, 'last');
            end
            if H1(r)
                H1 = false(p, 1);
                H1(r) = true;
                H2  = false(p, 1);
                fH1 = false(p, 1);
                fH1(r) = true; %%find(F == r);
                fH2 = false(p, 1);
            else
                H1  = false(p, 1);
                H2  = false(p, 1);
                H2(r) = true;
                fH1 = false(p, 1);
                fH2 = false(p, 1);
                fH2(r) = true;
            end
        end
    end
        %Fprime = F;
        F(fH1) = false;
        F(H2) = true;
        %%% only relevant for a p > n scenario.
        %if  length(F) > min(n,p)
        %    %%% remove indices such that F is small enough 
        %    lfH2 = min(n,p) - length(Fprime);
        %    [so, ix] = sort(lambda(H2));
        %    H2 = H2(ix(1:lfH2));
        %    F = [Fprime H2];
        %    Lambda = union(setdiff(Lambda, H2), H1);
        %else
        
        Lambda(H1) = true; 
        Lambda(H2) = false;
        % union(setdiff(Lambda, H2), H1);
        lfH2 = length(fH2);
        %end
        %%%
        
        lfH1 = length(fH1);
        
        %%% 0.1: rule of thumb.
        if  flag && (lfH1 + lfH2) <= max(1, 0.1 * length(F))
            % use up- and downdating
            % downdating, maintaining ordering
            if lfH1 > 0 
                for j=1:lfH1
                    dd = find(fH1, j, 'last');
                    R = choldownmatlab(R, dd(j));
                end
            end
            % then updating
            if lfH2 > 0
                for j=1:lfH2
                    R = cholinsertgram(R, AtA(H2(j), H2(j)), AtA(H2(j), Fprime));
                    Fprime(H2(j)) = 1;
                end
            end
        else
            R = chol(AtA(F,F));
            flag = 1;
        end
        x = zeros(p, 1);
        x(F) =  R \ (R' \ Atb(F));
        %%% mv
        if mv == 1
            lambda(Lambda) = AtA(Lambda, F) * x(F) - reshape(Atb(Lambda), sum(Lambda), 1);
        else
            lambda(Lambda) = transpose(A(:,Lambda)) * (A(:, F) * x(F) - b);
        end
        %%%
        lambda(F) = 0.9 * eps;
        %%%
        iter = iter + 1;
        %%%
        check = getkktopt(lambda, x);
        %%% gradient = lambda
        %perf = getL(options_general.perfmeas, lambda, x);
        %perf = perf(~isnan(perf));
        %ConvSpeed =  [ConvSpeed; [toc(t0) perf]];
        %check = check_termination(t0, options_general, perf, lambda, x, check.fprev, check.xprev); 
end

out.xopt = x;
%out.err = f(x);
%out.ConvSpeed = ConvSpeed; 
out.check = check;    

end


% Fast Cholesky insert and remove functions
function R = cholinsertgram(R, diag_k, col_k)
if isempty(R)
  R = sqrt(diag_k);
else
%  col_k = x'*X; % 
  R_k = R'\col_k'; % R'R_k = (X'X)_k, solve for R_k
  R_kk = sqrt(diag_k - R_k'*R_k); % norm(x'x) = norm(R'*R), find last element by exclusion
  R = [R R_k; [zeros(1,size(R,2)) R_kk]]; % update R
end
end



function Lkt =  choldownmatlab(Lt, k) 
% This is for non-sparse matrix
% cholesky downdating
% A in R^(n,p)
% G = A'* A = L * L', where L, L' come from cholesky decomposition
% now  removes kth column from A, denoted by Ak. Gk := Ak' * Ak
% Given L' and k, choldown computes the chol. decomposition of  Gk
% i.e. Lk' * Lk = Gk, without processing of A, G

p = length(Lt);

% drop the kth clm of Lt
Temp = Lt;
Temp(:,k) = []; % Temp in R^(p,p-1)

% Givens Rotations
for i = k:p-1,
    a = Temp(i,i);
    b = Temp(i+1,i);
    r = sqrt(sum(Lt(:,i+1).^2) - sum(Temp(1:i-1,i).^2));
    c =  r * a / (a^2+b^2);
    s =  r * b / (a^2+b^2);
    % ith row of rotation matrix H
    Hrowi = zeros(1,p); Hrowi(i) = c; Hrowi(i+1) = s; 
    % (i+1)th row of ration matrix H
    Hrowi1 = zeros(1,p); Hrowi1(i) = -s; Hrowi1(i+1) = c;
    % modify the ith and (i+1)th rows of Temp
    v = zeros(2,p-1);
    v(1,i:p-1) = Hrowi * Temp(:,i:p-1);
    v(2,i+1:p-1) = Hrowi1 * Temp(:,i+1:p-1);
    Temp(i:i+1,:) =  v;
end

% drop the last row
Lkt = Temp(1:p-1,:);
end


    