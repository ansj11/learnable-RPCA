function test_forward()

load('/home/ansj/Data/imdb.mat');
x = imdb.images.data(:,:,2,100)/255;
label = imdb.images.labels(:,:,:,100);
label = label/max(label(:));
x = x(:);
label = label(:);
aviobj = VideoWriter('Outlier.avi');
aviobj.FrameRate = 0.001;
load('/home/ansj/Data/D.mat');
f = 1;
lambda  = 1/255;
lambda1 = 0.01;
[~, q]  = size(D);

W = D' * f;  
H = eye(q) - (D' * D + lambda1 * eye(q))*f;  
t = lambda*f; 

s = zeros(32,1,'single');
o = zeros(183992,1,'single');
y = [];
for i = 1:5000
    s = H * s + W * (x - o);
    o = Threshold((1-f)*o+f*(x-D*s), t);
    
    [err_s,err_o] = Loss(s, o, x, D, label); 
    fprintf('iter %3d, err = %.0f. \n', i, err_o);
    y = [y err_o];
    
    im = 255*reshape(o,422,436);
    im = uint8(im/max(im(:)));
    open(aviobj);
    writeVideo(aviobj,im);
	close(aviobj);
end
