import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.covariance
import scipy
import os, pickle
from subprocess import Popen, PIPE
import Util
import uuid

def get_uuid():
    return uuid.UUID(bytes=os.urandom(16), version=4)

def equiweight_cov(X):
    n, p = X.shape
    return np.eye(p)

def MTP2_wrapper(X):
    cov = np.cov(X.T, bias=True)
    return MTP2_cov_wrapper(cov)

def MTP2_cov_wrapper(cov):
    og_dir = os.getcwd()
    try:
        os.chdir("./matlab")
        mdict = {'S': cov}
        uid = get_uuid()
        inp_path = './data/algo_sample_cov_{}.mat'.format(uid)
        out_path = './data/algo_est_{}.mat'.format(uid)
        scipy.io.savemat(inp_path, mdict)
        command = "matlab -nodisplay -nodesktop -r \"computeomega '{}' '{}'; exit;\"".format(inp_path, out_path)
        os.system(command)
        #make sure this waits for the above to finish?
        ans = scipy.io.loadmat(out_path)['Omega']
    finally:
        os.chdir(og_dir)
        os.remove(inp_path)
        os.remove(out_path)
    return np.linalg.inv(ans)

def glasso_wrapper(X):
    try:
        glasso_method = sklearn.covariance.GraphicalLassoCV(cv=3)
        glasso_method.fit(X)
        return np.linalg.inv(glasso_method.get_precision())
    except:
        return None

def tiger_wrapper(X):
    uid = get_uuid()
    X = X - np.mean(X, axis = 0)
    np.save(os.path.join(os.getcwd(), "rscripts", "tiger_in_{}.npy".format(uid)), X)
    args = ['Rscript', 'rscripts/tiger.R', str(uid)]
    p = Popen(args, stdout=PIPE)
    while p.poll() is None:
        print(p.stdout.readline())
    cov = np.load(os.path.join(os.getcwd(), "tiger_out_{}.npy".format(uid)))
    os.remove(os.path.join(os.getcwd(), "rscripts", "tiger_in_{}.npy".format(uid)))
    os.remove(os.path.join(os.getcwd(), "rscripts", "tiger_out_{}.npy".format(uid)))
    return np.linalg.inv(cov)

def POET_wrapper(X):
    uid = get_uuid()
    X = X - np.mean(X, axis = 0)
    in_name = os.path.join(os.getcwd(), "rscripts", "POET_in_{}.npy".format(uid))
    out_name = os.path.join(os.getcwd(), "rscripts", "POET_out_{}.npy".format(uid))
    np.save(in_name, X)
    args = ['Rscript', 'rscripts/POET_script.R', str(uid)]
    p = Popen(args, stdout=PIPE)
    while p.poll() is None:
        print(p.stdout.readline())
    cov = np.load(out_name)
    os.remove(in_name)
    os.remove(out_name)
    return cov

def POET_5_wrapper(X):
    uid = get_uuid()
    X = X - np.mean(X, axis = 0)
    in_name = os.path.join(os.getcwd(), "rscripts", "POET_in_{}.npy".format(uid))
    out_name = os.path.join(os.getcwd(), "rscripts", "POET_out_{}.npy".format(uid))
    np.save(in_name, X)
    args = ['Rscript', 'rscripts/POET_script_5facs.R', str(uid)]
    p = Popen(args, stdout=PIPE)
    while p.poll() is None:
        print(p.stdout.readline())
    cov = np.load(out_name)
    os.remove(in_name)
    os.remove(out_name)
    return cov

def NLS_wrapper(X):
    uid = get_uuid()
    X = X - np.mean(X, axis = 0)
    in_name = os.path.join(os.getcwd(), "rscripts", "NLS_in_{}.npy".format(uid))
    out_name = os.path.join(os.getcwd(), "rscripts", "NLS_out_{}.npy".format(uid))
    np.save(in_name, X)
    args = ['Rscript', 'rscripts/NLS.R', str(uid)]
    p = Popen(args, stdout=PIPE)
    while p.poll() is None:
        print(p.stdout.readline())
    NLS = np.load(out_name)
    os.remove(in_name)
    os.remove(out_name)
    return NLS

def old_LS_wrapper(X, cov=None):
    from linear_shrinkage import estimator
    return estimator(X)

def LS_wrapper(X):
    uid = get_uuid()
    X = X - np.mean(X, axis = 0)
    in_name = os.path.join(os.getcwd(), "rscripts", "LS_in_{}.npy".format(uid))
    out_name = os.path.join(os.getcwd(), "rscripts", "LS_out_{}.npy".format(uid))
    np.save(in_name, X)
    args = ['Rscript', 'rscripts/LS.R', str(uid)]
    p = Popen(args, stdout=PIPE)
    while p.poll() is None:
        print(p.stdout.readline())
    LS = np.load(out_name)
    os.remove(in_name)
    os.remove(out_name)
    return LS

def LRPS_wrapper(X):
    X = X - np.mean(X,axis=0)
    uid = get_uuid()
    in_name = os.path.join(os.getcwd(), "rscripts", "lrps_in_{}.npy".format(uid))
    out_name = os.path.join(os.getcwd(), "rscripts", "lrps_out_{}.npy".format(uid))
    np.save(in_name, X)
    args = ['Rscript', 'rscripts/lrps.R', str(uid)]
    p = Popen(args, stdout=PIPE)
    while p.poll() is None:
        print(p.stdout.readline())
    omega = np.load(out_name)
    os.remove(in_name)
    os.remove(out_name)
    return np.linalg.inv(omega)

def CLIME_wrapper(X):
    X = X - np.mean(X,axis=0)
    uid = get_uuid()
    in_name = os.path.join(os.getcwd(), "rscripts", "clime_in_{}.npy".format(uid))
    out_name = os.path.join(os.getcwd(), "rscripts", "clime_out_{}.npy".format(uid))
    np.save(in_name, X)
    args = ['Rscript', 'rscripts/clime.R', str(uid)]
    p = Popen(args, stdout=PIPE)
    output = []
    while p.poll() is None:
        lin = p.stdout.readline()
        output.append(lin)
        print(lin)
    output_str = [x.decode('utf-8') for x in output] 
    if 'Error in solve.default(Sigma)' in ''.join(output_str):
        return None
    omega = np.load(out_name)
    os.remove(in_name)
    os.remove(out_name)
    return np.linalg.inv(omega)

def CLIME_cov_wrapper(X, T):
    #T is the number of stocks
    assert X.shape[0] == X.shape[1] #to make sure it's a covariance matrix
    uid = get_uuid()
    in_name = os.path.join(os.getcwd(), "rscripts", "clime_cov_in_{}.npy".format(uid))
    out_name = os.path.join(os.getcwd(), "rscripts", "clime_cov_out_{}.npy".format(uid))
    np.save(in_name, X)
    args = ['Rscript', 'rscripts/clime_cov.R', str(uid), str(T)]
    p = Popen(args, stdout=PIPE)
    while p.poll() is None:
        print(p.stdout.readline())
    omega = np.load(out_name)
    os.remove(in_name)
    os.remove(out_name)
    return np.linalg.inv(omega)

def get_AFM_estimator(num_factors, residual_method, tradeidx):
    assert num_factors in [1, 5]
    FF_5 = np.load('data/FF_5.npy')
    FF_1 = FF_5[:, 0].reshape(-1, 1)
    if num_factors == 1:
        FF = FF_1
    elif num_factors == 5:
        FF = FF_5

    assert residual_method in ['LS', 'NLS']
    #load factor matrix and get the appropriate # of factors
    if residual_method == 'LS':
        residual_estimator = old_LS_wrapper
    elif residual_method == 'NLS':
        residual_estimator = NLS_wrapper
    else:
        assert False, "Invalid residual method"

    def AFM_estimator(pastRet, h):
        T, N = pastRet.shape
        fac_mat = Util.get_past_period_factor(h, T, tradeidx, FF)

        assert fac_mat.shape == (T, num_factors)
        B = []
        res_mat = [] 
        for i in range(N):
            lm = LinearRegression()
            asset = pastRet[:, i]
            lm.fit(X = fac_mat, y = asset)
            beta_i = lm.coef_
            B.append(beta_i)
            pred = lm.predict(X = fac_mat)
            res = asset - pred
            res_mat.append(res)
        res_mat = np.array(res_mat).T
        assert(res_mat.shape[0] == T)
        B = np.array(B).T
        S_f = np.cov(fac_mat.T, bias=True)

        return B.T.dot(S_f).dot(B) + residual_estimator(res_mat)
    
    return AFM_estimator