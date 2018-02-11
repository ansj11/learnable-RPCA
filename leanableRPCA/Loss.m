function dx = Loss(x, label, Dz)

P = label>0;
N = label<=0;
hinge = max(1-(P-N).*abs(x),0);

% forward process
if nargin == 2
    dx = sum(sum(hinge));
end

% back propagation
if nargin == 3
    dx = ((P-N).*abs(x)<1).*(N-P).*sign(x);
end
