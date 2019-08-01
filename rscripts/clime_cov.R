library(reticulate)
library(clime)
library(RcppCNPy)

args <- commandArgs(trailingOnly=TRUE)
print(args)
uid <- args[1]
n <- as.integer(args[2])
#uid <- "cb31ebe8-1c61-43d8-b51a-95a4a1164670"
#n <- 200
inp = sprintf("/Users/umaroy/Documents/meng/MTP2-finance/clime_cov_in_%s.npy", uid)

S <- npyLoad(inp, dotranspose = FALSE)
#this is the covariance matrix
print("Loaded S")
p <- NCOL(S)
lambda_opt <- sqrt(log10(p)/n)
print(lambda_opt)
climeres <- clime(S,lambda=lambda_opt,
                  sigma=TRUE,
                  standardize=FALSE,
                  perturb = FALSE)
print("Done with CLIME")
res <- climeres$Omegalist[[1]]

out = sprintf("/Users/umaroy/Documents/meng/MTP2-finance/clime_cov_out_%s.npy", uid)
npySave(out, res)



