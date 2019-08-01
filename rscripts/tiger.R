library(flare)
library(RcppCNPy)

uid = commandArgs(trailingOnly=TRUE)
inp = sprintf("/Users/umaroy/Documents/meng/MTP2-finance/tiger_in_%s.npy", uid)
X <- npyLoad(inp, dotranspose=FALSE)
print("Loaded X")
d <- NCOL(X)
n <- NROW(X)
opt_lambda <- sqrt(log(d)/n)
res <- sugm(data=X,
            method='tiger',
            standardize=FALSE,
            perturb=FALSE)

print("DONE with tiger")
cov<-res$icov[[1]]

out = sprintf("/Users/umaroy/Documents/meng/MTP2-finance/tiger_out_%s.npy", uid)
npySave(out, cov)