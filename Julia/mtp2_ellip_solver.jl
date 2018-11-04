using Convex

function MTP2EllipticalEstimator(corr_mat, lambda)
    p = size(corr_mat)[2] # Number of variables
    Omega = zeros((p, p))
    for i in 1:p
        B = Variable(p)
        off_diag_vars = [B[j] for j in 1:p if j != i]
        constraint1 = maximum(corr_mat * B - eye(p)[:, i]) <= lambda 
        constraint2 = off_diag_vars .<= 0
        problem = minimize(vecnorm(B, 1))
        problem.constraints += constraint1
        problem.constraints += constraint2
        solve!(problem)
        Omega[:, i] = B.value
    end
    return Omega
end