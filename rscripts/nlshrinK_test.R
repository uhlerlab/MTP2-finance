library(nlshrink)
library(reticulate)
library(tictoc)
np <- import("numpy", convert=FALSE)
test_X <- np$load('/Users/umaroy/Documents/meng/MTP2-finance/test_X.npy')
n <- NROW(test_X) #sample size

test_X <- matrix(rexp(500*1000, rate=.1), ncol=500)
NROW(test_X)
NCOL(test_X)
tic('hi')
res <- linshrink_cov(test_X, 1)
toc()

tic("hi")
res2 <- nlshrink_cov(test_X, k=1)
toc()

n <- 100
p <- 50
test_X <- matrix(rnorm(n*p), nrow=n)
library(clime)
tic()
clime_obj <- clime(test_X)
toc()
clime_cv<-cv.clime(clime_obj)
clime_cv_opt <- clime(test_X, clime_cv$lambdaopt)
toc()

library(LRpS)
tic()
gam <- 0.1
path <- cross.validate.low.rank.plus.sparse(test_X,
                                            gamma = gam,
                                            n = n,
                                            covariance.estimator = Kendall.correlation.estimator,
                                            n.lambdas = 6,
                                            verbose = T,
                                            seed = 1)
best.ft <- choose.cross.validate.low.rank.plus.sparse(path)
S <- best.ft$fit$S
L <- best.ft$fit$L
Sigma.hat <- S + L
toc()
library(RcppCNPy)
tic()
test <- npyLoad("/Users/umaroy/Documents/meng/MTP2-finance/fmat.npy")
toc()
