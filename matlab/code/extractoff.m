function [off] = extractoff(A)

%%% function that extracts all distinct off-diagonal entries 
%%% of a symmetric matrix A (and returns a vector) 

mask = logical(triu(1 - eye(size(A,1)), 1));

Amask = A(mask); 

off = Amask(:);


end

