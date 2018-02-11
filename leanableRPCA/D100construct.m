load('/home/ansj/dataSet/D100.mat');
load('/home/ansj/dataSet/imdb151.mat');
%load('/home/ansj/dataSet/images3110.mat');

%%construct
for i=1:size(images,4)
    im = images(:,:,:,i);
    X = im(:);
    C = D'*X;
    Y = D *C;
    h = figure(i);
    subplot(2,2,1),imagesc(reshape(Y,422,436));
    subplot(2,2,2),imagesc(im)
    subplot(2,2,3),imagesc(reshape(X-Y,422,436));
    set(h,'visible','off');
    str = sprintf('/home/ansj/RPCA/D100construct/fig-%d',i);
    saveas(h,str,'jpg')
end