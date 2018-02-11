%% normal images
load('/home/ansj/dataSet/images3110.mat');
load('/home/ansj/dataSet/D100.mat');
im = images(:,:,:,50);
X = im(:);
C = D'*X;
Y = D*C;
load('/home/ansj/dataSet/B.mat');
cols = im2colstep(double(im),[15 15],[1 1]);
v = U'*cols;
V = U*v;
vessel = cols2im(V,422,436,1);
figure,
subplot(1,2,1),imagesc(reshape(Y+vessel,422,436));
subplot(1,2,2),imagesc(im)

load('/home/ansj/dataSet/imdb151.mat');
im = images(:,:,:,50);
X = reshape(im,183992,1);
C = D'*X;
Y = D*C;
cols = im2colstep(double(im),[15 15],[1 1]);
v = U'*cols;
V = U*v;
vessel = cols2im(V,422,436,1);
figure,
subplot(1,2,1),imagesc(reshape(Y+vessel,422,436));
subplot(1,2,2),imagesc(im)