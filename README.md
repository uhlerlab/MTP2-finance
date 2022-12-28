# MTP2-finance

MTP2 for covariance estimation in financial data.

# Data instructions

Create folder 'data/' and download data from here: https://drive.google.com/file/d/1smSlLg8K7RAvWm-EEDukHLxvaGmxnhqy/view?usp=sharing
The file `data_details.txt` has a description of the various different csvs and how they're relevant.

# Files

## Estimators

`estimators.py` contains the code for the various covariance estimators that we benchmark in our paper. 

## Utilities

`Util.py` is used for various utility functions, such as reading in the data, calculating relevant metrics, etc.

## Sample Usage

The iPython notebook `Generate Portfolios.ipynb` gives a example walk-through of how one would use an covariance estimator on the relevant data to calculate a portfolio of returns in the desired out of sample period.

## Matlab Instructions
Note that the MTP2 estimator requires matlab installed on your machine. You need to be able to run `matlab` from the command line, as that is how the estimator is called. To do this, you may need to add `export PATH=$PATH:/Applications/MATLAB_R2018a.app/bin` (with whichever version of `matlab` you have) to your `~/.bash_profile`. Verify the `matlab` interpreter comes up when you type `matlab` into your command line.

# References

* Raj Agrawal, Uma Roy, Caroline Uhler,
[Covariance Matrix Estimation under Total Positivity for Portfolio Selection](https://doi.org/10.1093/jjfinec/nbaa018),
Journal of Financial Econometrics, Volume 20, Issue 2, Spring 2022, Pages 367–389.
