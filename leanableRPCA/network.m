function [res,binerr] = network(net, x, label, dz)
%NETNN evaluate a neural network 

N = length(net.layers);
res = struct(...
    'x',cell(1,N+1),...
    'dx',cell(1,N+1),...
    'dw',cell(1,N+1));

%% forward process
for n=1:N
    l = net.layers{n};
    switch l.type
        case 'DTX'
			res(n+1).x = l.weights{1}*x;
        case 'DX'
            res(n+1).x = x-l.weights{1}*res(n).x;
        case 'BTX'
            res_x = im2colstep(reshape(res(n).x,422,436),[21 21],[1 1]);
            res(n+1).x = l.weights{1}*res_x;
        case 'relu'
            res(n+1).x = vl_nnrelu(res(n).x);
        case 'BX'
            res(n+1).x = cols2im(res_x-l.weights{1}*res(n).x,[21 21],[422,436],[1 1]);
        case 'threshold'
            res(n+1).x = Threshold(res(n).x, l.weights{1});
        case 'loss'
            res(n+1).x = Loss(res(n).x,label);
        otherwise
            break;
    end
end                 %

% backward propagation
if ~isempty(dz)
    res(N+1).dx = dz;
    for n=N:-1:1 % 2
        l = net.layers{n};
        switch l.type
            case 'loss'
                res(n).dx = Loss(res(n).x, label, res(N+1).dx); 
            case 'threshold'
                [res(n).dx, res(n).dw{1}] = Threshold(res(n).x, l.weights{1}, res(n+1).dx);
            case 'BX'
                tmp_dx = im2colstep(reshape(res(n+1).dx,422,436),[21 21],[1 1]);
                res(n).dx = -l.weights{1}'*tmp_dx;
                res(n).dw{1} = tmp_dx*res(n).x';
            case 'relu'
                res(n).dx = vl_nnrelu(res(n).x,res(n+1).dx);
            case 'BTX'
                res(n).dx = cols2im(l.weights{1}'*res(n+1).dx,[21 21],[422,436],[1 1]);
                res(n).dw{1} = res(n+1).dx*res(n).x';
            case 'DX'
                res(n).dx = -l.weights{1}'*(res(n+1).dx+tmp_dx);
                res(n).dw{1} = (res(n+1).dx+tmp_dx)*res(n).x';
            case 'DTX'
                res(n).dw{1} = res(n+1).dx*res(n).x';
            otherwise
                break;
        end

    end

end
prediction = (abs(res(end-1).x)>=1).*(label>0);
binerr = sum(prediction) / sum(label(:)>0) ;

