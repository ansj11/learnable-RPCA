eps = randn(422,436,'single');
x = eps(:) + X(:,1);

y = D*(D'*x);

figure,
subplot(1,2,1),imagesc(reshape(x,422,436));
title('noise image')
subplot(1,2,2),imagesc(reshape(y,422,436));
title('denoise image')

