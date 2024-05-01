function [x_min, y_min, T_min] = maximizer_MT( C, mu, theta, tau, X_min, X_max)
%MAXIMIZER_MT Outputs the point (first coordinate always being 1) that maximizes T according 
%to MATLAB's own functions    
    options = optimset('Display','off','MaxFunEvals',1e8, "TolFun", 1e-8);
    %The problem has no linear constraints, so set those arguments to [].
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    % For y = 1
    T_pos = @(x)(tau*norm([1;x])^2 -2)*[1;x]'*C*[1;x] +...
                2*(mu' - theta')*(1 - mu'*[1;x])*[1;x]+ ...
                tau*norm([1;x])^2 * (1 - mu'*[1;x])^2;
    % For y = -1
    T_neg = @(x)(tau*norm([1;x])^2 -2)*[1;x]'*C*[1;x] +...
                2*(mu' - theta')*(-1 - mu'*[1;x])*[1;x]+ ...
                tau*norm([1;x])^2 * (-1 - mu'*[1;x])^2;
    x0 = rand(length(mu)-1, 1); 
    % Compute argmax of T with y=-1 and y=1, among both points choose the
    % one that maximizes T.
    [x_min_pos, fval_pos] = fmincon(T_pos,x0,A, b, Aeq, beq, ...
        X_min*ones(length(mu)-1, 1), X_max*ones(length(mu)-1, 1), [],  options);
    [x_min_neg, fval_neg] = fmincon(T_neg,x0,A, b, Aeq, beq, ...
        X_min*ones(length(mu)-1, 1), X_max*ones(length(mu)-1, 1), [],  options);
    if (fval_neg < fval_pos)
        x_min = [1;x_min_neg];
        y_min = -1;        
        T_min = fval_neg;
    else
        x_min = [1;x_min_pos];
        y_min = 1;
        T_min = fval_pos;
    end    
end