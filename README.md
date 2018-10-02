# MTP2-finance
MTP2 for covariance estimation in financial data

# Setup Instructions

Create folder 'data/' and download data from here: https://www.dropbox.com/sh/zj7lxs25hsn2osi/AACx0aoUATK53eYvNCrNsDR0a?dl=0

# Description

The prelim.ipynb contains the preliminary work to load in the data

# Notes from Wolff

Original data is here: https://www.dropbox.com/s/fsyepfq3o9g4anm/Caroline.mat?dl=0 (converted to .csv for ease of use with Python)

## Suggestions (from email)

Enclosed is a working paper with some relevant portfolios. I would suggest you do an analysis similar to Table 1 on page 15 but using the following portfolios only:
1/N: as described in the paper
Lin: like NL in the paper but use the linear shrinkage estimator Ledoit and Wolf (2004, JMVA)
MTP: like NL in the paper, but use your estimator of the covariance matrix — or the shrinkage estimator where your estimator is the target
AFM1-LIN: like AMF1-NL in the paper, but use the linear shrinkage estimator on the residuals
AFM1-MTP: like AMF1-NL in the paper, but use your estimator of the covariance matrix — or the shrinkage estimator where your estimator is the target — on the residuals

## Explanation of non-constant weights on portfolio for OOS testing: 

Note that if you use daily data but update only every month (that is, every 21 days), then you should keep number of shares fixed during that month instead of the vector of portfolio weights, since the latter approach would incur daily trading due to the different developments of stock prices from day to day.

## Further explanation from email:

As I said before, these are data used in Section 6 of the attached paper.

There we used T = 1250 for the estimation of a covariance matrix, but that was for the dynamic DCC-NL model. Since you will be using static models instead (that is, models that assume i.i.d. data), you might want to use T = 250 or T = 500 instead for the estimation of a covariance matrix.
