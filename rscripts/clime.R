library(reticulate)
library(clime)
library(RcppCNPy)

new_dir <- paste(getwd(), "/rscripts", sep="")
setwd(new_dir)

uid = commandArgs(trailingOnly=TRUE)
inp = sprintf("clime_in_%s.npy", uid)

X <- npyLoad(inp, dotranspose=FALSE)
print("Loaded X")

n <- NROW(X)
p <- NCOL(X)
lambda_opt <- sqrt(log10(p)/n)

#Below is for cross-validated CLIME
# clime_obj <- clime(X,nlambda=10)
# print("Performed clime")
# clime_cv<-cv.clime(clime_obj,fold=3)
# print("Performed CV")
# print(clime_cv$lambdaopt)
# print(lambda_opt)
# clime_cv_opt <- clime(X, clime_cv$lambdaopt)
# res <- clime_cv_opt$Omegalist[[1]]

n <- NROW(X)
p <- NCOL(X)
lambda_opt <- sqrt(log(p)/n)
print(lambda_opt)
climeres <- clime(X,
               lambda=lambda_opt,
               sigma=FALSE,
               standardize=FALSE,
               perturb = FALSE)
res <- climeres$Omegalist[[1]]

out = sprintf("clime_out_%s.npy", uid)
npySave(out, res)


