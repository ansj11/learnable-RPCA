function [D, S] = dict(train)
% seek the dictionary D


% [U, S, ~] = svd(train);
% S = sqrt(S);
% D = U * S;
% D= D(:, 1:200);

% economy size decomposition
% 
% train     = double(train);
[U, S, ~] = svd(train, 0);
S         = sqrt(S);
D         = U * S;
% ������normalize

leg = sqrt(sum(D.^2));
D   = bsxfun(@rdivide, D, leg);

end

