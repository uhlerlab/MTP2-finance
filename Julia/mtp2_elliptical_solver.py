import julia
import pandas as pd
import numpy as np

try:
	j
	print("Julia already defined")
except:
	print("Julia not defined, loading solver...")
	j = julia.Julia()
	solver = j.include('mtp2_ellip_solver.jl')

def MTP2CovEstimator(data, lam):
	data = pd.DataFrame(data)
	kendal_corr_mat = data.corr(method='kendall').values
	corr_mat = np.sin(.5 * np.pi * kendal_corr_mat)
	print("Correlation matrix in MTP2CovEstimator function")
	print(corr_mat)
	return solver(corr_mat, lam)