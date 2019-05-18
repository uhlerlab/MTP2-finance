import numpy as np
from sklearn.linear_model import LinearRegression
import linear_shrinkage

def estimator(pastRet, fac_mat):
	T, N = pastRet.shape
	assert(fac_mat.shape[0] == T)
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
	S_f = np.cov(fac_mat.T)

	return B.T.dot(S_f).dot(B) + linear_shrinkage.estimator(res_mat)