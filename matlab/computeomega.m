function []=computeomega(arg1, arg2)
    addpath(genpath('./code'))
    
%     tic()
    load(arg1, 'S');
    %potentially could get this to just read a CSV?
    %reading a 256 by 256 mat takes 0.004 seconds
    %reading a 256 by 256 CSV takes 0.02 seconds
%     toc()
%     
%     tic()
%     S = csvread('testSout.csv');
%     toc()
    
    [Omega, ~, ~] = blockdescent_omega(S);
%     tic()
%     csvwrite(arg2, Sigma); %Takes 0.03 seconds for 256x256
%     toc()
    
%    tic()
    save(arg2, '-v7', 'Omega'); %This is not that much faster than above, takes 0.02 seconds%
%    toc()
end

