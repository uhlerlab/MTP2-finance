import pandas as pd
import numpy as np
import os
from os.path import join
import scipy.io
from multiprocessing import Pool
import linear_shrinkage
import factor_models
from subprocess import Popen, PIPE

def skip_weak(N, T):
    if N == 1000:
        return True
    if N == 500 and T == 1260:
        return True
    return False

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

def matlab_command_wrapper(string):
	ogcwd = os.getcwd()
	os.chdir('./matlab')
	os.system(string) #Have to make call within matlab folder, because that's where file is defined
	os.chdir(ogcwd)

def populate_cov_relax(folder, uid, pastRet, lamb):
	sampleCov = np.cov(pastRet.T)
	sampleCov -= lamb
	p, _ = sampleCov.shape
	sampleCov += np.eye(p) * lamb #to make sure that we're not subtracting from diag
	mdict = {'S': sampleCov}
	scipy.io.savemat(join("matlab", "data", folder, "{}.mat".format(uid)), mdict)

def populate_relax_covs_from_samples(folder, uids, pastRets, lamb):
	for u, r in zip(uids, pastRets):
		populate_cov_relax(folder, u, r, lamb)
	print("Done populating all relax covs")

def kendall_cov(data):
	df = pd.DataFrame(data)
	kendall_corr_mat = df.corr(method='kendall').values
	corr_mat = np.sin(0.5 * np.pi * kendall_corr_mat)
	stdmat = np.diag(np.sqrt(np.diag(np.cov(data.T))))
	return stdmat.dot(corr_mat).dot(stdmat)

def populate_cov_kendall(folder, uid, pastRet):
	sampleCov = kendall_cov(pastRet)
	mdict = {'S': sampleCov}
	scipy.io.savemat(join("matlab", "data", folder, "{}.mat".format(uid)), mdict)

def populate_kendall_covs_from_samples(folder, uids, pastRets):
	for u, r in zip(uids, pastRets):
		populate_cov_kendall(folder, u, r)
	print("Done populating all Kendall covs")

def populate_cov(folder, uid, pastRet = None, sampleCov = None):
	if pastRet is not None:
		sampleCov = np.cov(pastRet.T)
	mdict = {'S': sampleCov}
	scipy.io.savemat(join("matlab", "data", folder, "{}.mat".format(uid)), mdict)

def populate_covs(folder, uids, covs):
	#populate covs in folder with the given uids and covs
	#the covs are not calculated on the fly, they're already given
	for u, c in zip(uids, covs):
		populate_cov(folder, u, sampleCov = c)
	print("Done populating all covs")

def populate_covs_from_samples(folder, uids, pastRets):
	#populate covs by calcuating them from the samples (pastRets)
	for u, r in zip(uids, pastRets):
		populate_cov(folder, u, pastRet = r)
	print("Done populating all covs")

def save_AFM_out(data):
	arr = data['arr']
	data_folder = data['data_folder']
	prs = data['pastRets']
	pfs = data['pastFactors']

	for u, pastRet, pastFac in zip(arr, prs, pfs):
		est = factor_models.AFM1_LS(pastRet, pastFac)
		mdict = {'Sigma': est}
		scipy.io.savemat(join(data_folder, "AFM1LS_out_{}.mat".format(u)), mdict)

def run_parallel_AFM(cores, folder, pastRets, uids, pastFactors):
	data_folder = join("matlab", "data", folder)
	arrs = np.array_split(np.array(uids), cores)
	split_PRs = np.array_split(np.array(pastRets), cores)
	split_PFs = np.array_split(np.array(pastFactors), cores)
	datas = []

	for arr, pr, pf in zip(arrs, split_PRs, split_PFs):
		d = {}
		d['arr'] = arr
		d['data_folder'] = data_folder
		d['pastRets'] = pr
		d['pastFactors'] = pf
		datas.append(d)

	for d in datas:
		save_AFM_out(d)
		
	# with Pool(cores) as p:
	# 	p.map(save_AFM_out, datas)

def save_LS_out(data):
	arr = data['arr']
	data_folder = data['data_folder']
	prs = data['pastRets']

	for u, pastRet in zip(arr, prs):
		est = linear_shrinkage.estimator(pastRet)
		mdict = {'Sigma': est}
		scipy.io.savemat(join(data_folder, "LS_out_{}.mat".format(u)), mdict)

def run_parallel_LS(cores, folder, pastRets, uids):
	data_folder = join("matlab", "data", folder)
	arrs = np.array_split(np.array(uids), cores)
	split_PRs = np.array_split(np.array(pastRets), cores)
	datas = []

	for arr, pr in zip(arrs, split_PRs):
		d = {}
		d['arr'] = arr
		d['data_folder'] = data_folder
		d['pastRets'] = pr
		datas.append(d)

	#print(datas)
	#return datas

	for d in datas:
		save_LS_out(d)

	#with Pool(cores) as p:
	#	p.map(save_LS_out, datas)

	print("Done with running all commands for LS!")

def run_parallel_MTP2(cores, folder, uids):
	commands = []
	for i, arr in enumerate(np.array_split(np.array(uids), cores)):
		all_commands = []
		for u in arr:
			in_arr = "{}.mat".format(u)
			out_arr = "MTP2_out_{}.mat".format(u)
			data_folder = join("data", folder) #since called from matlab folder
			c = construct_command(in_arr, out_arr, data_folder = data_folder)
			all_commands.append(c)
		commands.append(construct_string_commands(all_commands))
	with Pool(cores) as p:
		p.map(matlab_command_wrapper, commands)

	print("Done with running all commands!")

def cumulative_variance(rets):
	vs = []
	for i in range(1, len(rets)):
		vs.append(np.var(rets[:i]))
	return vs

def cumulative_annualized_std(rets):
	stds = []
	for i in range(1, len(rets)):
		stds.append(100*np.sqrt(12)*np.std(rets[:i]))
	return stds

def load_EWTQ_OOS(folder):
	fname = join('matlab/data', folder, 'EWTQ_OOS_momentum_rets.npy')
	return np.load(fname)

def load_equiweight_OOS(folder):
	fname = join('matlab/data', folder, 'equiweight_OOS_rets.npy')
	return np.load(fname)

def load_OOS(folder, P = None, AFM = False, momentum = False):
	if momentum:
		mom = "momentum_"
	else:
		mom = ""
	if P:
		MTP2_fname = join('matlab/data', folder, 'MTP2_OOS_{}rets_P_{}.npy'.format(mom, P))
		if os.path.isfile(MTP2_fname):
			MTP2 = np.load(MTP2_fname)
		else:
			MTP2 = np.zeros(360)
		LS_fname = join('matlab/data', folder, 'LS_OOS_{}rets_P_{}.npy'.format(mom, P))
		if os.path.isfile(LS_fname):
			LS = np.load(LS_fname)
		else:
			LS = np.zeros(360)
	else:
		MTP2_fname = join('matlab/data', folder, 'MTP2_OOS_{}rets.npy'.format(mom))
		if os.path.isfile(MTP2_fname):
			MTP2 = np.load(MTP2_fname)
		else:
			MTP2 = np.zeros(360)
		LS_fname = join('matlab/data', folder, 'LS_OOS_{}rets.npy'.format(mom))
		if os.path.isfile(LS_fname):
			LS = np.load(LS_fname)
		else:
			LS = np.zeros(360)
		if AFM:
			AFM_fname = join('matlab/data', folder, 'AFM1LS_OOS_{}rets.npy'.format(mom))
			if os.path.isfile(AFM_fname):
				AFM = np.load(AFM_fname)
			else:
				AFM = np.zeros(360)

	return MTP2, LS, AFM

def load_losses(folder, P = None):
	if P:
		MTP2 = np.load(join('matlab/data', folder, 'MTP2_OOS_losses_P_{}.npy'.format(P)))
		LS = np.load(join('matlab/data', folder, 'LS_OOS_losses_P_{}.npy'.format(P)))	
	else:
		MTP2 = np.load(join('matlab/data', folder, 'MTP2_OOS_losses.npy'))
		LS = np.load(join('matlab/data', folder, 'LS_OOS_losses.npy'))
	return MTP2, LS

def get_LS_OOS_custom_univ(folder, P, subset, tradeidx, ret, P_in_name = False):
	rets = []
	losses = []
	for h in range(360-P):
		LS_cov = get_LS_cov(folder, h)
		w = optimal_weights(LS_cov)
		outRet = get_invest_period_custom_univ(h, P, subset, tradeidx, ret)
		curret = retConstShare(outRet, w)
		rets.append(curret)
		S = np.cov(outRet.T)
		try:
			np.linalg.inv(S)
		except:
			print("The sample covariance matrix is singular!")

		l = loss(LS_cov, S)
		losses.append(l)

	if P_in_name:
		rets_name = 'LS_OOS_rets_P_{}'.format(P)
		losses_name = 'LS_OOS_losses_P_{}'.format(P)
	else:
		rets_name = 'LS_OOS_rets'
		losses_name = 'LS_OOS_losses'

	np.save(join('matlab/data/', folder, rets_name), rets)
	np.save(join('matlab/data/', folder, losses_name), losses)
	return rets, losses

def get_LS_OOS(folder, N, P, univ, tradeidx, ret):
	#assumes the uids are index into universe
	rets = []
	for h in range(len(univ)):
		LS_cov = get_LS_cov(folder, h)
		w = optimal_weights(LS_cov)
		outRet = get_invest_period(h, P, N, univ, tradeidx, ret)
		curret = retConstShare(outRet, w)
		rets.append(curret)

	np.save(join('matlab/data/', folder, 'LS_OOS_rets'), rets)
	return rets

def get_equiweight_OOS(folder, N, P, univ, tradeidx, ret):
	rets = []
	for h in range(len(univ)):
		w = np.ones(N) / N
		outRet = get_invest_period(h, P, N, univ, tradeidx, ret)
		curret = retConstShare(outRet, w)
		rets.append(curret)
	np.save(join('matlab/data/', folder, 'equiweight_OOS_rets'), rets)
	return rets

def get_AFM_OOS(folder, N, P, univ, tradeidx, ret):
	#assumes the uids are index into universe
	rets = []
	for h in range(len(univ)):
		LS_cov = get_AFM_cov(folder, h)
		w = optimal_weights(LS_cov)
		outRet = get_invest_period(h, P, N, univ, tradeidx, ret)
		curret = retConstShare(outRet, w)
		rets.append(curret)

	np.save(join('matlab/data/', folder, 'AFM1LS_OOS_rets'), rets)
	return rets

def get_MTP2_OOS_custom_univ(folder, P, subset, tradeidx, ret, P_in_name = False):
	rets = []
	losses = []
	for h in range(360-P):
		MTP2_cov = get_MTP2_cov(folder, h)
		w = optimal_weights(MTP2_cov)
		outRet = get_invest_period_custom_univ(h, P, subset, tradeidx, ret)
		curret = retConstShare(outRet, w)
		rets.append(curret)
		S = np.cov(outRet.T)
		try:
			np.linalg.inv(S)
		except:
			print("The sample covariance matrix is singular!")

		l = loss(MTP2_cov, S)
		losses.append(l)

	if P_in_name:
		rets_name = 'MTP2_OOS_rets_P_{}'.format(P)
		losses_name = 'MTP2_OOS_losses_P_{}'.format(P)
	else:
		rets_name = 'MTP2_OOS_rets'
		losses_name = 'MTP2_OOS_losses'

	np.save(join('matlab/data/', folder, rets_name), rets)
	np.save(join('matlab/data/', folder, losses_name), losses)
	return rets, losses

def get_momentum_OOS(folder, N, P, univ, tradeidx, ret, cov_func, method_name):
	rets = []
	for h in range(len(univ)):
		cov = cov_func(folder, h)
		m = get_momentum_signal(h, N, univ, tradeidx, ret)
		b = np.mean(m)
		w = optimal_weights_momentum(m, cov, b)
		outRet = get_invest_period(h, P, N, univ, tradeidx, ret)
		curret = retConstShare(outRet, w)
		rets.append(curret)

	np.save(join('matlab/data/', folder, '{}_OOS_momentum_rets'.format(method_name)), rets)
	return rets

def get_EWTQ_OOS(folder, N, P, univ, tradeidx, ret):
	rets = []
	for h in range(len(univ)):
		m = get_momentum_signal(h, N, univ, tradeidx, ret)
		perc = np.percentile(m, 80)
		top = m >= perc
		tot = int(sum(top))
		w = top / tot
		outRet = get_invest_period(h, P, N, univ, tradeidx, ret)
		curret = retConstShare(outRet, w)
		rets.append(curret)
	np.save(join('matlab/data/', folder, 'EWTQ_OOS_momentum_rets'), rets)
	return rets

def get_MTP2_momentum_OOS(folder, N, P, univ, tradeidx, ret):
	get_momentum_OOS(folder, N, P, univ, tradeidx, ret, get_MTP2_cov, 'MTP2')

def get_AFM_momentum_OOS(folder, N, P, univ, tradeidx, ret):
	get_momentum_OOS(folder, N, P, univ, tradeidx, ret, get_AFM_cov, 'AFM1LS')

def get_LS_momentum_OOS(folder, N, P, univ, tradeidx, ret):
	get_momentum_OOS(folder, N, P, univ, tradeidx, ret, get_LS_cov, 'LS')


def get_MTP2_OOS(folder, N, P, univ, tradeidx, ret):
	#assumes the uids are index into universe
	rets = []
	for h in range(len(univ)):
		MTP2_cov = get_MTP2_cov(folder, h)
		w = optimal_weights(MTP2_cov)
		outRet = get_invest_period(h, P, N, univ, tradeidx, ret)
		curret = retConstShare(outRet, w)
		rets.append(curret)

	np.save(join('matlab/data/', folder, 'MTP2_OOS_rets'), rets)
	return rets

def get_AFM_cov(folder, uid):
	data_folder = join('matlab/data', folder)
	return scipy.io.loadmat(format_appro('AFM1LS_out_{}.mat'.format(uid), data_folder))['Sigma']

def get_LS_cov(folder, uid):
	data_folder = join('matlab/data', folder)
	return scipy.io.loadmat(format_appro('LS_out_{}.mat'.format(uid), data_folder))['Sigma']

def get_MTP2_cov(folder, uid):
	data_folder = join('matlab/data', folder)
	return scipy.io.loadmat(format_appro('MTP2_out_{}.mat'.format(uid), data_folder))['Sigma']

def MTP2_cov(N, T):
	return lambda h: scipy.io.loadmat(format_appro('out_N_{}_T_{}_unividx_{}.mat'.format(N, T, h), 'matlab/data'))['Sigma']

def MTP2_cov_folder(folder, s_num, T):
	return lambda h: scipy.io.loadmat(os.path.join(folder, 'out_subset_{}_T_{}_unividx_{}.mat'.format(s_num, T, h)))['Sigma']

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

def get_past_period_custom_univ(h, T, subset, tradeidx, ret):
	today = tradeidx[h][0]
	pastPeriod = range(today - T, today)
	pastRet = ret[pastPeriod][:, subset]
	return pastRet

def get_invest_period_custom_univ(h, P, subset, tradeidx, ret):
	today = tradeidx[h][0]
	investPeriod = range(today, today + P*21)
	outRet = ret[investPeriod][:, subset]
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

def loss(Sigma_hat, Sigma):
	#up to a factor of N up front
	K_hat = np.linalg.inv(Sigma_hat)
	K = np.linalg.inv(Sigma)
	num = np.trace(K_hat.dot(Sigma).dot(K_hat))
	denom = np.trace(K_hat)**2
	return num/denom - 1./np.trace(K)

def get_std(rets):
	return get_IR(rets)[1]

def get_IR(rets):        
	avg = 100 * 12 * np.mean(rets)
	std = 100 * np.sqrt(12)*float(np.std(rets))
	if std == 0:
		return 0, 0, 0
	else:
		return avg, std, avg/std

def hypothesis_testing(rets1, rets2, folder, name1, name2, sharpe):
	hyp_mat = np.vstack((rets1, rets2)).T
	hyp_mat_name = "{}_vs_{}.npy".format(name1, name2)
	in_file = join(os.getcwd(), 'matlab/data', folder, hyp_mat_name)
	if sharpe:
		method = 'sharpe'
	else:
		method = 'var'
	out_file = join(os.getcwd(), 'matlab/data', folder, '{}_vs_{}_res_{}.csv'.format(name1, name2, method))
	np.save(in_file, hyp_mat)
	args = ['rscript', 'p_values.R', in_file, out_file]
	if sharpe: 
		args.append('sharpe')
	p = Popen(args, stdout=PIPE)
	output = p.stdout.read()
	print(output)

def read_hyp_results(folder, name1, name2, sharpe):
	if sharpe:
		method = 'sharpe'
	else:
		method = 'var'
	out_file = join(os.getcwd(), 'matlab/data', folder, '{}_vs_{}_res_{}.csv'.format(name1, name2, method))
	df = pd.read_csv(out_file)
	return df
#####################################################################

#This is for back when we were worried about nans
#nan_perc = np.isnan(pastRet).sum(axis = 0)  / T
#nan_thres = nan_perc < 0.0025 #only include if less than 2.5% is nan
#new_universe = universe[nan_thres]
#print("Number in new universe: {} out of {}".format(new_universe.shape[0], N))
#pastRet = ret_nonan[pastPeriod][:, new_universe]
 
def cov_to_corr(cov):
	cov_diag = np.diag(np.power(np.diag(cov), -0.5))
	return cov_diag.dot(cov).dot(cov_diag)     

def check_positive(cov):
	total = 0
	violation = 0
	for i in range(len(cov)):
		for j in range(i+1, len(cov)):
			total += 1
			if cov[i][j] < 0:
				violation += 1
	return violation / total

def check_MTP2(cov):
	K = np.linalg.inv(cov)
	total = 0
	violation = 0
	violation_sum = 0
	for i in range(len(K)):
		for j in range(i+1, len(K)):
			total += 1
			if K[i][j] > 0:
				violation += 1
				violation_sum += K[i][j]
	return violation / total, violation_sum / violation

def subsample(rets, T = 50, frac = 0.5):
	#Given rets, bootstrap stds
	S = int(len(rets) * frac) #how much to include in each subsample
	stds = []
	for t in range(T):
		subset = np.random.choice(rets, size = S)
		stds.append(np.std(subset) * np.sqrt(252 / 20))
	return stds