%% visible
im = reshape(res(end-1).y,422,436);
ll = reshape(label,422,436);
figure,
subplot(1,2,1),imagesc(im);
subplot(1,2,2),imagesc(ll);
max(im(:))


%% test learning rate
for i = 1:numel(net.layers)
    if isfield(net.layers{i},'weights')
        for j  = 1:numel(net.layers{i}.weights)
            tmp = bsxfun(@rdivide,res(i).dw{j},net.layers{i}.weights{j}+0.0000001);
            rate(i,j) = max(tmp(:));
        end
    end
end

%% test gradient

im = reshape(res(end-1).dy,422,436);
im_forward = reshape(res(end-1).y,422,436);
figure,
subplot(1,2,1),imagesc(im);
subplot(1,2,2),imagesc(im_forward);