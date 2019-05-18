#library(RcppCNPy)
load("~/Documents/meng/MTP2-finance/R_packages/Var.RData")
load("~/Documents/meng/MTP2-finance/R_packages/Sharpe.RData")
library(reticulate)
np <- import("numpy")
args = commandArgs(trailingOnly=TRUE)
ret <- np$load(args[1])
stopifnot(NCOL(ret) == 2)

if (length(args) >= 3) {
    sharpe <- TRUE
} else {
    sharpe <- FALSE
}

if (sharpe) {
    print("Calculating for Sharpe!")
    res1 <- hac.inference(ret)
    res2 <- boot.time.inference(ret, 10, 499)
} else {
    print("Calculating for variance!")
    res1 <- hac.inference.log.var(ret)
    res2 <- boot.time.inference.log.var(ret, 10, 499)
}
#print(res1$Difference)
#print(res1$p.Values)
#print(res2$Difference)
#print(res2$p.Value)

diff <- res2$Difference
HAC.pValue <- res1$p.Values[['HAC']]
HAC.pw.pValue <- res1$p.Values[['HAC.pw']]
bootstrap.pValue <- res2$p.Value

arr <- list("difference"=diff, 
            "HAC"=HAC.pValue, 
            "HAC_pw"=HAC.pw.pValue, 
            "bootstrap"=bootstrap.pValue)
write.csv(arr, "", row.names=FALSE)
write.csv(arr, args[2], row.names=FALSE)
