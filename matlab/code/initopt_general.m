function options = initopt_general(varargin)

%function options = initopt_general(xstar, maxcpu, stopcrit, tol, usegram, gram, mv, mode)
% Input
%   perfmeas        - perfmeans measure
%   maxcpu          - maximum cputime
%   stopcrit        - stopping criterion
%   tol             - tolerance
%   usegram         - whether to use gram matrix
%   gram            - precomputed Gram matrix
%   mv              - How to do the matrix vector-multiplications for
%                     evaluations of the gradient
%   mode            - Input mode.
%                       0: Inputs A and b are used directly.
%                       1: NNLS problem in KKT form is solved, i.e. Input
%                           A is A' * A,
%                           b is A' * b.
%                      Note: '1' makes sense only if usegram=1

psr =  inputParser;

validPerfmeas = @(x)iscell(x) && ~isempty(x) && length(x)<=3;
psr.addParamValue('perfmeas', {1}, validPerfmeas); % name, default, validator

validMaxcpu = @(x)validateattributes(x,{'numeric'},{'scalar','positive'});
psr.addParamValue('maxcpu',inf,validMaxcpu);

validStopcrit = @(x)validateattributes(x,{'numeric'},{'vector','positive','integer','<=',3});
psr.addParamValue('stopcrit',1,validStopcrit);

validTol = @(x)validateattributes(x,{'numeric'},{'vector','positive'});
psr.addParamValue('tol',10^-10,validTol);

validUsegram = @(x)validateattributes(x,{'numeric'},{'scalar','binary'});
psr.addParamValue('usegram',0,validUsegram);

validGram = @(x)validateattributes(x,{'numeric'},{'2d'});
psr.addParamValue('gram',[],validGram);

validMv = @(x)validateattributes(x,{'numeric'},{'scalar','integer','nonnegative','<=',2});
psr.addParamValue('mv',0,validMv);

validMode = @(x)validateattributes(x,{'numeric'},{'scalar','integer','nonnegative','<=',1});
psr.addParamValue('mode',0,validMode);

% 
% options.perfmeas = {[1], [2], xstar}; % potentially xstar = []
% options.maxcpu  = maxcpu; % may be infinite
% options.stopcrit = stopcrit;
% if ~all(size(stopcrit) == size(tol))
%    error('Dimension of stopcrit does not match the dimension of tol') 
% end
% options.tol = tol;
% options.usegram = usegram;
% options.gram = gram;
% options.mv = mv;
% options.mode = mode;

% Parse and validate all input arguments.
psr.parse(varargin{:});
options = psr.Results;

options.stopcrit = unique(options.stopcrit);
if length(options.tol)~= length(options.stopcrit)
    error('The length of tolerance vector should be the same as the length of stopping criterions.');
end

if ~isempty(options.gram)
    options.usegram = true;
end

if  options.usegram == 0
    
    %options = rmfield(options,'gram');
    if  options.mode == 1
        % error('mode 1 makes sense only if usegram = 1.');
        display('mode 1: automatically set usegram = 1.')    % you can decide what you want to do, giving an error, changing the setting, etc.
        options.usegram = 1;
    end
    
    if options.mv == 1
        % error('mv 1 makes sense only if usegram = 1.');
        display('mv 1: automatically set usegram = 1.')
        options.usegram = 1;
    end
    
end

% example
% initopt_general()
% initopt_general('usegram',true)
% initopt_general('gram',ones(3,3));
% initopt_general('mv',2,'tol',10^-8);