using Convex

function MTP2CovEstimator(data)
    N = size(data)[1] # Number of samples
    p = size(data)[2] # Number of variables
    cov_mat = (transpose(data) * data) / N
    K = Semidefinite(p, p)
    off_diag_vars = [K[i, j] for i in 1:p for j in 1:p if i > j] # Entries above diagonal            
    problem = maximize(logdet(K) - trace(K * cov_mat), off_diag_vars .<= 0)
    solve!(problem)
    return inv(K.value)
end