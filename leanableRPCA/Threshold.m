function [y, Dw] = Threshold(x, t, Dz)    
 

 
 TH = abs(t);
 B = x >= TH;     % if X(i) >= TH(i), then B(i) = 1, else = 0
 S = x <= -TH;    % if X(i) <= -TH(i), then S(i) = 1, else = 0

 % forward process
 if nargin == 2
     y  = (x - TH) .* B + (x + TH) .* S;
 end
 
 % back propogation
 if nargin == 3
     y = Dz .* (B + S); 
     Dw = mean(mean(Dz .* (-B + S).*sign(t)));    % softmax derivation with T  / m*n
 
 end
  
end