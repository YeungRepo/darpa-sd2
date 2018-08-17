function [dx ] = lin_hybrid(t,x, A1, A2)
if x(1) < .05
    out = A1 * x;
else
    out = A2 * x;
end
x1_prime = out(1);
x2_prime = out(2);
dx = [x1_prime;x2_prime];
end
