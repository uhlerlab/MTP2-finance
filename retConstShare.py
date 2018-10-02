# % INPUTS:  retMat = n x p matrix of (simple) returns
# %          w = p x 1 vector of portfolio weights
# %          inPerc = indicator variable of whether the returns are in percent 
# % ------------------------------------------------------------------
# % RETURNS: retVec = n x 1 vector of portfolio returns 
# %                   (in percent or not, same as input)
# %          wEnd = weights of assets at end of investment period

#assumes retMat is not in percent

n, p = retMat.shape
assert(w.shape == (p,1))

wSum1 = w/np.sum(w)

totalRetMat = 1 + retMat

cummProdd = np.cumprod(totalRetMat)
navVec = np.matmul(cummProdd, wSum1)

wEnd = cummProdd[n, :]
wEnd = np.dot(wEnd, w.T) #since w is (p,1) but wEnd is (1,p)
wEnd = wEnd/np.sum(wEnd)
wEnd = wEnd.T

totalRetVec = np.divide(navVec, np.concatenate([1], navVec[1:(n-1)]))

retVec = totalRetVec - 1
retVec = retVec * np.sum(w)

return retVec
