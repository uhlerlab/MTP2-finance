import julia

j = julia.Julia()
solver = j.include('mtp2_cov_solver.jl')

def MTP2CovEstimator(data):
    return solver(data)
