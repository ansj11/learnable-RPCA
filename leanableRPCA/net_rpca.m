function [net,res,binerr] = rpca_net(net, x, label, res, dz)
%NETNN evaluate a neural network 

N = length(net.layers);
res = struct(...
    'x',cell(1,N+1),...
    'y',cell(1,N+1),...
    'dx',cell(1,N+1),...
    'dy',cell(1,N+1),...
    'dw',cell(1,N+1));
res(1).x = x;

%% forward process
for n=1:N
    l = net.layers{n};
    switch l.type
        case 'init'
			res(n+1).x = net.layers{1}.weights{2}*res(n).x;
            res(n+1).y = net.layers{1}.weights{4}*res(n).x;
        case 'iter'
            res(n+1).x = net.layers{1}.weights{1}*res(n).x + net.layers{1}.weights{2}*(res(1).x-res(n).y) ;
            res(n+1).y = net.layers{1}.weights{3}*res(n).y + net.layers{1}.weights{4}*res(1).x - net.layers{1}.weights{5}*res(n).x ;
        case 'soft'
            res(n+1).x = res(n).x ;
            res(n+1).y = Threshold(res(n).y, l.weights{1});
        case 'loss'
            [res(n+1).x,res(n+1).y] = Loss(res(n).x,res(n).y,x,l.weights{1},label);
        otherwise
            break;
    end;
end;                 %

% backward propagation
if ~isempty(dz)
    res(N+1).dx  = dz;
    res(1).dw{1} = 0;res(1).dw{2} = 0;res(1).dw{3} = 0;res(1).dw{4} = 0;res(1).dw{5} = 0;
    for n=N:-1:1 % 2
        l = net.layers{n};
        switch l.type
            case 'loss'
                [res(n).dx, res(n).dy,res(n).dw{1}] = Loss(res(n).x, res(n).y,...
                    x,l.weights{1}, label, res(N+1).dx); 
            case 'soft'
                res(n).dx = res(n+1).dx;
                [res(n).dy, res(n).dw{1}] = Threshold(res(n).y, l.weights{1}, res(n+1).dy);
            case 'iter'
                res(n).dx = net.layers{1}.weights{1}'*res(n+1).dx - net.layers{1}.weights{5}'*res(n+1).dy;
                res(n).dy = net.layers{1}.weights{3}'*res(n+1).dy - net.layers{1}.weights{2}'*res(n+1).dx;
                res(1).dw{1} = res(1).dw{1} + res(n+1).dx*res(n).x';
                res(1).dw{2} = res(1).dw{2} + res(n+1).dx*(res(1).x - res(n).y)';
                res(1).dw{3} = res(1).dw{3} + mean(res(n+1).dy .* res(n).y);
                res(1).dw{4} = res(1).dw{4} + mean(res(n+1).dy .* res(1).x);
                res(1).dw{5} = res(1).dw{5} - res(n+1).dy * res(n).x';
            case 'init'
                res(n).dx = net.layers{1}.weights{1}' * res(n+1).dx;
                res(n).dy = net.layers{1}.weights{4}' * res(n+1).dy;
                res(1).dw{2} = res(1).dw{2} + res(n+1).dx*res(n).x';
                res(1).dw{4} = res(1).dw{4} + mean(res(n+1).dy .* res(n).x);
            otherwise
                break;
        end
    end
end
prediction = (abs(res(end-1).y)>=1).*(label>0);
binerr = sum(prediction) / sum(label(:)>0) ;
clear prediction