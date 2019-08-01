library(LRpS)
library(reticulate)
library(RcppCNPy)

uid = commandArgs(trailingOnly=TRUE)
inp = sprintf("/Users/umaroy/Documents/meng/MTP2-finance/lrps_in_%s.npy", uid)

ret <- npyLoad(inp, dotranspose=FALSE)
n <- NROW(ret) #sample size
set.seed(0)
#lambda(gamma * |S|_1 + (1-gamma)*Tr(L))
gam <- 0.15
#path <- fit.low.rank.plus.sparse.path(Sigma=Sigma, 
#                                      gamma = gam,
#                                      n = n,
#                                      n.lambdas = 40)
path <- cross.validate.low.rank.plus.sparse(ret,
                                            gamma = gam,
                                            n = n,
                                            #covariance.estimator = cor,
                                            n.folds=3,
                                            covariance.estimator = Kendall.correlation.estimator,
                                            n.lambdas = 10,
                                            verbose = TRUE,
                                            lambda.ratio=1e-03,
                                            seed = 1)

best.ft <- choose.cross.validate.low.rank.plus.sparse(path)
S <- best.ft$fit$S
L <- best.ft$fit$L
Sigma.hat <- S + L

out = sprintf("/Users/umaroy/Documents/meng/MTP2-finance/lrps_out_%s.npy", uid)
npySave(out, Sigma.hat)
print("Saved result")