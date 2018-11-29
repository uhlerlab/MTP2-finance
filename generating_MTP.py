import numpy as np
#Block to generate the matrices

def assert_symm(mat):
    assert(np.linalg.norm(mat-mat.T) == 0)

def assert_PSD(cov):
    try:
        np.linalg.cholesky(cov)
    except:
        print("Not PSD!")

def get_adj(inv):
    adj = (inv != 0).astype(int)
    return adj

def get_S(adj):
    S = np.max(np.sum(adj, axis = 0)) - 1
    return S

def generate_mat(p, target_S, sparse_thres = 0.7):
    #p is the number of nodes, i.e. dimensionality of the matrix
    #S is goal maximum degree of the graph
    #sparse_thres is a tuning parameter that should be adjusted depending on S
    #@returns the inv of the covariance matrix (precision matrix)
    S = target_S - 1
    count = 0
    while S != target_S:
        #print(count)
        count += 1
        if S > target_S:
            sparse_thres *= 1.15
        else:
            sparse_thres *= 0.85
        sam_inv = np.zeros((p,p))
        for i in range(p):
            for j in range(i+1, p):
                if np.random.uniform() < sparse_thres:
                    sam_inv[i][j] = 0
                else:
                    sam_inv[i][j] = np.random.uniform(low = -1., high = -0.2) 

        inv = np.minimum(sam_inv, sam_inv.T) #makes it symmetric
        adj = get_adj(inv + np.eye(p)) #for factoring in the diagonal to adjacency matrix calculation
        S = get_S(adj)

    for i in range(p): #sets it to be diagonally dominant
        s = abs(np.sum(inv[i]))
        inv[i][i] = s + np.random.uniform(low = 0.2, high = 1.)
    
    adj = get_adj(inv)
    S = get_S(adj)
    assert_symm(inv)
    #print(S)
    assert(S == target_S)
    return inv