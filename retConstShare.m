function [retVec,wEnd] = retConstShare(retMat,w,inPerc)
% PURPOSE: computes returns for portfolio that keeps number of shares fixed
% ------------------------------------------------------------------
% INPUTS:  retMat = n x p matrix of (simple) returns
%          w = p x 1 vector of portfolio weights
%          inPerc = indicator variable of whether the returns are in percent 
% ------------------------------------------------------------------
% RETURNS: retVec = n x 1 vector of portfolio returns 
%                   (in percent or not, same as input)
%          wEnd = weights of assets at end of investment period
% ------------------------------------------------------------------
% NOTES:   allows for the possibility of the weights not summing up one
% ------------------------------------------------------------------

% written by: Michael Wolf
% CREATED  06/14
% UPDATED 

[n,p] = size(retMat);
if (length(w) ~= p)
    error('dimensions of retMat and w do not match');
end
[a,b] = size(w);
if (1 == a)
    w = w';
end

wSum1 = w/sum(w);

% convert simple returns to total returns
if inPerc
    totalRetMat = 1+retMat./100;
else
    totalRetMat = 1+retMat;
end

% vector of NAVs of portfolio over time
cummProdd = cumprod(totalRetMat);
navVec = cummProdd*wSum1;

% figure out vector of weights at the end of investment period
wEnd = cummProdd(n,:);
wEnd = wEnd.*w';
wEnd = wEnd/sum(wEnd);
wEnd = wEnd';

% convert NAVs to total portfolio returns
totalRetVec = navVec./[1;navVec(1:(n-1))];

% convert total portfolio returns to simple portfolio returns
if inPerc
    retVec = (totalRetVec-1).*100;
else 
    retVec = totalRetVec-1;
end
    
retVec = retVec*sum(w);
   

      






