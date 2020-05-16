import pandas as pd
import numpy as np
import os
from os.path import join
import scipy.io
from multiprocessing import Pool
from subprocess import Popen, PIPE


def kendall_cov(data):
	df = pd.DataFrame(data)
	kendall_corr_mat = df.corr(method='kendall').values
	corr_mat = np.sin(0.5 * np.pi * kendall_corr_mat)
	stdmat = np.diag(np.sqrt(np.diag(np.cov(data.T))))
	return stdmat.dot(corr_mat).dot(stdmat)

def print_normalize(a):
	print(np.mean(a) * 252 / 20, np.std(a) * np.sqrt(252 / 20))

# Utilities for dealing with financial data

def load_data():
	ret = pd.read_csv('data/ret.csv', header = None).values 
	#ret.shape = (10344, 3251), (day, stock)
	univ = pd.read_csv('data/topMV95.csv', header = None).values 
	univ -= 1 #because Matlab is 1 indexed
	#univ.shape = (360, 1000), (OOS month, sorted list of stocks to consider for that period)
	dates = pd.read_csv('data/mydatestr.txt', header = None, parse_dates = [0]) 
	#dates.shape = (10344, 1), date for each day in return (not a numpy array, but a dataframe with DT objects)
	tradeidx = pd.read_csv('data/investDateIdx.csv', header = None).values 
	tradeidx -= 1 #because Matlab is 1 indexed!
	#tradeidx.shape = (360, 1), (row of univ -> index in ret matrix)
	ret[ret == -500] = np.nan
	ret = ret / 100 #ret is in percent
	ret_nonan = ret.copy()
	ret_nonan[np.isnan(ret)] = 0

	return ret, ret_nonan, univ, tradeidx, dates

def get_past_period(h, T, N, univ, tradeidx, ret):
	universe = univ[h,:N]
	today = tradeidx[h][0]
	pastPeriod = range(today-T, today)
	pastRet = ret[pastPeriod][:, universe]
	return pastRet

def get_past_period_factor(h, T, tradeidx, FF):
	today = tradeidx[h][0]
	pastPeriod = range(today-T, today)
	return FF[pastPeriod]

def get_invest_period(h, P, N, univ, tradeidx, ret):
	#P is the lookahead, in months
	universe = univ[h,:N]
	today = tradeidx[h][0]
	investPeriod = range(today, today + P*21)
	outRet = ret[investPeriod][:, universe]
	return outRet

def get_momentum_period(h, N, univ, tradeidx, ret):
	universe = univ[h,:N]
	today = tradeidx[h][0]
	investPeriod = range(today - 252, today - 21)
	outRet = ret[investPeriod][:, universe]
	return outRet

def get_momentum_signal(h, N, univ, tradeidx, ret):
	rets = get_momentum_period(h, N, univ, tradeidx, ret)
	rets += 1.
	return scipy.stats.gmean(rets)
	
# Utilities for dealing with covariance computation and OOS computations

def sample_cov_nonan(pastRet):
	#Computes sample covariance when there are nans
	df = pd.DataFrame(pastRet)
	clean_cov = df.cov() #does this by removing nans
	if np.count_nonzero(np.isnan(clean_cov)) > 0:
		print("{} nans in covariance matrix".format(np.count_nonzero(np.isnan(clean_cov))))
		assert False
	return clean_cov

def identity_cov(pastRet):
	_, N = pastRet.shape
	return np.eye(N)

def getCumRet_sanity_check(outret):
	#Given out of sample returns, sanity checks calculation of cumret vector
	#The cumret vector is used in OOS_rets before
	cum_rets = []
	D, N = outret.shape #D is days, N is number of stocks
	for i in range(N):
		stock_ret = 1.
		for d in range(D):
			day_ret = outret[d, i]
			stock_ret *= (1+day_ret) #day_ret should NOT be in percent
		cum_rets.append(stock_ret - 1.) #to convert back to percentage increase
	return cum_rets #N for each stock

def OOS_rets(outret, w):
	#This is if you don't want to keep # of shares constant 
	#It is the 'pure' version of retConstShare below
	assert np.isclose(np.sum(w), 1.0)
	total_ret = outret + 1
	cum_ret = np.cumprod(total_ret, axis=0)[-1,:]
	cum_ret -= 1
	return np.dot(w, cum_ret)

def retConstShare(retMat, w):
	n, p = retMat.shape
	if len(w.shape) == 1:
		w = np.expand_dims(w, 1)
	assert(w.shape == (p,1))
	wSum1 = w/np.sum(w)

	totalRetMat = 1 + retMat

	cummProdd = np.cumprod(totalRetMat, axis = 0)
	navVec = np.matmul(cummProdd, wSum1)

	wEnd = cummProdd[n-1, :]
	wEnd = np.dot(wEnd, w) #since w is (p,1) but wEnd is (1,p)
	wEnd = wEnd/np.sum(wEnd)
	wEnd = wEnd.T

	navVecTot = np.concatenate((np.ones((1,1)), navVec[:(n-1),]))

	totalRetVec = np.divide(navVec, navVecTot)

	retVec = totalRetVec - 1
	retVec = retVec * np.sum(w)

	return np.sum(retVec) #sum of all of the returns

def optimal_weights(cov):
	n = cov.shape[0]
	prec = np.linalg.inv(cov)
	denom = np.matmul(np.matmul(np.ones(n), prec), np.ones(n))
	return np.matmul(prec, np.ones(n)) / denom

def optimal_weights_momentum(m, sigma, b):
	n = sigma.shape[0]
	prec = np.linalg.inv(sigma)
	A = np.matmul(np.matmul(np.ones(n), prec), np.ones(n))
	B = np.matmul(np.matmul(np.ones(n), prec), np.ones(n) * b)
	C = np.matmul(np.matmul(m, prec), m)
	c_1 = (C - b*B) / (A*C - B**2)
	c_2 = (b*A - B) / (A*C - B**2)
	w = c_1 * np.matmul(prec, np.ones(n)) + c_2 * np.matmul(prec, m)
	return w

def get_std_None_safe(rets, verbose=True):
	if None in rets:
		li = []
		for x in rets:
			if x != None:
				li.append(x)
		if verbose:
			print("# of none null is:", len(li), "Out of:", len(rets))
		return get_std(li)
	else:
		return get_std(rets)

def get_IR_None_safe(rets, verbose=True):
	if None in rets:
		li = []
		for x in rets:
			if x != None:
				li.append(x)
		if verbose:
			print("# of none null is:", len(li), "Out of:", len(rets))
		return get_IR(li)[2]
	else:
		return get_IR(rets)[2]

def get_std(rets):
	return get_IR(rets)[1]

def get_IR(rets):   
	avg = 100 * 12 * np.mean(rets)
	std = 100 * np.sqrt(12)*float(np.std(rets))
	if std == 0:
		return 0, 0, 0
	else:
		return avg, std, avg/std