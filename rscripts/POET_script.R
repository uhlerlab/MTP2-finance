library(POET)
library(RcppCNPy)
new_dir <- paste(getwd(), "/rscripts", sep="")
setwd(new_dir)

uid = commandArgs(trailingOnly=TRUE)
inp = sprintf("POET_in_%s.npy", uid)
X <- npyLoad(inp, dotranspose=FALSE)
print("Loaded X")
Y <- t(X)

# K<-POETKhat(Y)$K1HL
# print("Determined K")
# print(K)

res<-POET(Y,3)
print("Done with POET")
cov<-res$SigmaY

out = sprintf("POET_out_%s.npy", uid)
npySave(out, cov)