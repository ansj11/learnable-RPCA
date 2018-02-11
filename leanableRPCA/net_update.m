function net = net_update(net, res, LearnRate, mt, wd, n)


%% use decay learning rate
for i = 1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
        for j = 1:numel(net.layers{i}.weights)
            net.learnrate{i}{j} = LearnRate/ceil(n/5);
        end
    end
end

%% Normalize gradient and incorporate weight decay.
for i = 1:numel(net.layers)
	l = net.layers{i};
    if isfield(net.layers{i}, 'weights')
        for j = 1:numel(l.weights)
            
            % Normalize gradient and incorporate weight decay.
            V = wd*net.layers{i}.weights{j} + res(i).dw{j}; % V = W*wd + DzDw　
            net.momentum{i}{j} = mt*net.momentum{i}{j} - net.learnrate{i}{j}*V ;			 % mu = mu - V
            net.layers{i}.weights{j} = net.layers{i}.weights{j} + net.momentum{i}{j} ;
            % W = W + mu = W + mt*mu - lr*(W*wd - DzDw)　
            
        end
    end
end

						   