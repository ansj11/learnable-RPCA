function net = net_train()

%% opts
EpochNumber = 1000;        % Number of updation
saveEvery    = 10;           
LearnRate     = 1e-6;        % Step size
Momentum    = 0.9;         % 
WeightDecay = 0.000005;
clip = 0.1;

%% net setup
load('/home/ansj/dataSet/imdb151.mat');
imdb = load('/home/ansj/dataSet/imdb32.mat');
path = '/home/ansj/RPCA/out';
modelFigPath = fullfile(path, 'net-train.jpg');
modelPath = @(ep) fullfile(path, sprintf('net-epoch-%d.mat', ep));

%% training
stats = [];
start = findLastCheckpoint(path) ;
if start >= 1
    fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
    load(modelPath(start)) ;
else
	net = net_init();
end
for en = start+1:EpochNumber
	err = [];
    tic;
    for k = 1:151
        x = images(:,:,:,15);
        label = labels(:,:,:,15);
        dz = 1;
        res = [];
        [res,binerr] = network(net, x(:), label(:), dz);           
        err = [err [res(end).x; binerr]] ; 	%计算的是batch的平均误差
		stats.train(:,k) = sum(err,2)/k ;
        fprintf('Epoch %3d, iter %3d, err %.6f, binerr %.3f. ', en, k, stats.train(1,k),stats.train(2,k));
        toc; 
        if 0 
            eps = 0.01; index = 1;
            i = 1;j = 1
            net.layers{i}.weights{j}(id) = net.layers{i}.weights{j}(id) + eps;
            [ck, ~] = network(net,x(:), label(:), dz);
            dw = (ck(end).x - res(end).x)/eps;
            fprintf("Calculate grad %.3f, check grad %.3f\n",res(i).dw{j}(id),dw);
        end
        net = net_update(net, res, LearnRate, Momentum, WeightDecay, en);
    end
    err = [];
    tic;
    for k = 1:32
		x = imdb.images(:,:,:,k);
        label = imdb.labels(:,:,:,k);
        dz = [];
        res= [];
        [res,binerr] = netNN(net, x(:), label(:), dz);           
        err = [err [res(end).y; binerr]] ;
		stats.valid(:,k) = sum(err,2)/(k-151) ;
        fprintf('Epoch %3d, iter %3d, err %.2f, binerr %.3f. ', en, k, stats.valid(1,k),stats.valid(2,k));
        toc; 
    end
    if ~mod(en,saveEvery)
        save(modelPath(k), 'net', 'stats');
    end
	h = figure;
	subplot(1,2,1),plot(1:en,stats.train(1,:),'b-',1:en,stats.valid(1,:),'r-');
	title('Hinge Loss')
	subplot(1,2,2),plot(1:en,stats.train(2,:),'b-',1:en,stats.valid(2,:),'r-');
	title('Binary error')
	set(h,'visible','on');
	saveas(h,modelFigPath)
	
end
figure, 
subplot(2,2,1), imagesc(res(end-1).x);
title('outlier');
subplot(2,2,2), imagesc(label);
title('label');
subplot(2,2,3), imagesc(res(end-1).y)
title('ground image');



% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
