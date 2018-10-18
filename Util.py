import pandas as pd
import numpy as np
import os

###
#Nonsense for Matlab
###
def format_appro(f, data_folder = 'data', ext = '.mat'):
	'''Given a filename
	--formats it with the full-path of the data_folder
	--formats it to end with '.mat'
	Example: testS -> ./data/testS.mat
	'''
	if not '/' in f:
		f = os.path.join('./' + data_folder, f)
	if not f.endswith('.mat'):
		f = f + '.mat'
	return f

def construct_command(f, o, data_folder = 'data/'):
	''' Given an input and output file name, 
	writes command for computing covariance with those parameters'''
	f = format_appro(f, data_folder)
	o = format_appro(o, data_folder)
	return "computecov '{}' '{}'".format(f, o)

def construct_string_commands(clist):
	'''Given a list of covariance commands
	strings them together as a command-line command to run matlab'''
	string = "matlab -nodisplay -nodesktop -r \""
	for c in clist:
		string += "{}; ".format(c)
	string += "exit;\""
	return string

#####################################################################

def print_normalize(a):
    print(np.mean(a) * 252 / 20, np.std(a) * np.sqrt(252 / 20))


###
#Utilities for dealing with financial data
###
#####################################################################

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

def get_invest_period(h, P, N, univ, tradeidx, ret):
	#P is the lookahead, in months
	universe = univ[h,:N]
	today = tradeidx[h][0]
	investPeriod = range(today, today + P*21)
	outRet = ret[investPeriod][:, universe]
	return outRet

###
# Utilities for dealing with covariance computation and OOS computations
###
#####################################################################

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

#####################################################################

#This is for back when we were worried about nans
#nan_perc = np.isnan(pastRet).sum(axis = 0)  / T
#nan_thres = nan_perc < 0.0025 #only include if less than 2.5% is nan
#new_universe = universe[nan_thres]
#print("Number in new universe: {} out of {}".format(new_universe.shape[0], N))
#pastRet = ret_nonan[pastPeriod][:, new_universe]
        
