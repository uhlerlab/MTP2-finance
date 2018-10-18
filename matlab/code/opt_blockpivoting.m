function options = opt_blockpivoting(varargin)
  
p = inputParser;

validdown = @(x)validateattributes(x,{'numeric'},{'scalar','integer','positive','<=',2});
p.addParamValue('down',1,validdown);

validitmax = @(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'});
p.addParamValue('itmax', inf, validitmax);

validqbar = @(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'});
p.addParamValue('qbar', 10, validqbar);

p.parse(varargin{:});
options = p.Results;

% 
% options.down = down;
% options.itmax = itmax;
% options.qbar = qbar;
% 
