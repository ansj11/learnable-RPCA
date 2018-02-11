function net = net_init()

f = 1/100;
lambda = 1/255;
beta   = 0.01;
load('/home/ansj/dataSet/D100.mat');

q = size(D,2);
H = (1-beta*f)*eye(q) - f*D'*D;
h = 1-f;
W = f*D';
t = lambda*f*ones(1,3,'single');
T = lambda*ones(1,1,128,'single');
BT= f*randn(15,15,3,128,'single');
B = 

net.layers = {};   

net.layers{end+1} = struct( ...
    'type',   'proj',...
    'weights',{{W}});

net.layers{end+1} = struct( ...
    'type',   'recon',...
    'weights',{{D}});

net.layers{end+1} = struct( ...
    'type',   'conv',...
    'weights',{{f*randn(128,441)}});

net.layers{end+1} = struct( ...
    'type',   'relu');

net.layers{end+1} = struct( ...
    'type',   'BX',...
    'weights',{{f*randn(128,441)}});

net.layers{end+1} = struct( ...
    'type',   'soft',...
    'weights',{{t}});

net.layers{end+1} = struct( ...
    'type',   'loss');

%% initialize momentum
    for i = 1:numel(net.layers)
        if isfield(net.layers{i}, 'weights')
            for j = 1:numel(net.layers{i}.weights)
                net.momentum{i}{j}  = 0 ; % initial momentum = 0
            end
        end
    end
end
