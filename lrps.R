library(LRpS)
library(reticulate)
np <- import("numpy")
#args = commandArgs(trailingOnly=TRUE)
#ret <- np$load(args[1])
ret <- np$load('/Users/umaroy/Documents/meng/MTP2-finance/testRet.npy')
n <- NROW(ret) #sample size
set.seed(0)
#lambda(gamma * |S|_1 + (1-gamma)*Tr(L))
gam <- 0.1
#path <- fit.low.rank.plus.sparse.path(Sigma=Sigma, 
#                                      gamma = gam,
#                                      n = n,
#                                      n.lambdas = 40)
path <- cross.validate.low.rank.plus.sparse(ret,
                                            gamma = gam,
                                            n = n,
                                            covariance.estimator = Kendall.correlation.estimator,
                                            n.lambdas = 3,
                                            verbose = T,
                                            seed = 1)

best.ft <- choose.cross.validate.low.rank.plus.sparse(path)
S <- best.ft$fit$S
L <- best.ft$fit$L
Sigma.hat <- S + L
np$save('/Users/umaroy/Documents/meng/MTP2-finance/TestSave.npy', Sigma.hat)

