import numpy as np

def inner_prod(A,B):
    p_n, _ = A.shape
    return 1./p_n * np.trace(A.dot(B.T))

def norm(A):
    return inner_prod(A,A)

def estimator(X):
    #be careful to make sure X is 0 mean?

    X = X - np.mean(X, axis = 0)
    S = np.cov(X.T)
    n, p = X.shape
    m = inner_prod(S, np.eye(p))
    d = norm(S - m * np.eye(p))

    bar_b = 0
    for x in X:
        xk = np.reshape(x, (p,1))
        bar_b += norm(xk.dot(xk.T) - S)
    bar_b = bar_b / n**2
    b = min(bar_b, d)
    a = d - b

    S_hat = b/d*m*np.eye(p) + a/d*S
    #print(b/d*m, a/d)
    return S_hat

def lambda_identity(lamb):
    return lambda X: shrink_iden(X, lamb)

def shrink_iden(X, lamb):
    S = np.cov(X.T)
    n, p = X.shape
    return lamb * np.eye(p) + (1-lamb) * S