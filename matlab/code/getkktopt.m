function[kktopt] = getkktopt(grad, x)

num = abs(grad) .* (grad > eps);
score = num;

Pc = num < eps;

if sum(Pc) > 0
    score(Pc) = num(Pc);
end

if sum(~Pc) > 0
   score(~Pc) = num(~Pc) ./ x(~Pc);
end

Pc = num < eps | score < 1;
P  = ~Pc;

if sum(P) > 0
   kktP = max(abs(x(P)));
else
   kktP = 0;
end

if sum(Pc) > 0
   kktPc = max(abs(grad(Pc)));
else
   kktPc = 0; 
end

N = x < -eps;

if sum(N) > 0
   kktN = max(abs(x(N)));
else
   kktN = 0; 
end

kktopt = max([kktP kktPc kktN]);


%O = ~N & x < eps;
%minx = min(x(x >= 0));

% P = x < eps; % this concerns the methods where the boundary can be reached.
% if any(P > 0)
%     if any(N > 0)
%        kktP = max(max(max(-min(grad(P), 0))), max(abs(x(N))));  
%     else    
%        kktP =  max(max(-min(grad(P), 0)));
%     end
% else
%     kktP = 0; 
% end
% if any(~P > 0)%.* x(~P)
%     kktPc_1  = max(grad(~P) .* (grad(~P) > 0));
%     kktPc_2  = max(abs(grad(~P)) .* (grad(~P) < 0));
%     kktPc = max([kktPc_1 kktPc_2]);
% else
%     kktPc = 0;
% end
% kktopt = max(kktP, kktPc);
end