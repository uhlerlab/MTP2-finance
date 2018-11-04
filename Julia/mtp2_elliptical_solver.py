import julia
import pandas as pd
import numpy as np

j = julia.Julia()
solver = j.include('mtp2_ellip_solver.jl')

def MTP2CovEstimator(data, lam):
	data = pd.DataFrame(data)
	kendal_corr_mat = data.corr(method='kendall').values
	corr_mat = np.sin(.5 * np.pi * kendal_corr_mat)
    return solver(corr_mat, lam)
