function [x_min, T_min] = probing_LfO( C, mu, X_min, X_max, sig)
%MAXIMIZER LfO Outputs the point (first coordinate always being 1) that minimizes
%the uncertainty (norm of covariance) according to MATLAB's own functions    
    options = optimset('Display','off','MaxFunEvals',1e15, "TolFun", 1e-15);
    %The problem has no linear constraints, so set those arguments to [].
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    T_prob = @(x)norm((eye(3)- (C * [1;x] * [1;x]')/([1;x]'*C*[1;x] + sig))*C);
    x0 = rand(length(mu)-1, 1); 
    [x_min_pos, T_min] = fmincon(T_prob,x0,A, b, Aeq, beq, X_min*ones(length(mu)-1, 1), X_max*ones(length(mu)-1, 1), [],  options);
    x_min = [1;x_min_pos];
end
