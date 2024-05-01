function [idx_min, y_min, min_values] = maximizer_predefinedPool( C, mu, theta, tau, words_pool)
%MINIMIZER Outputs the point of the pool that maximizes T, i.e. minimizes MSE from one
%iteration to the next.
    temp_pos = (tau -2)*sum(words_pool' * C .* words_pool', 2)' +... 
                 2 * (mu' - theta')*(words_pool .* (1 - mu'*words_pool)) + ...
                tau * (1 - mu'*words_pool).^2;
    temp_neg = (tau -2)*sum(words_pool' * C .* words_pool', 2)' +...
                 2 * (mu' - theta')*(words_pool .* (-1 - mu'*words_pool)) + ...
                tau * (-1 - mu'*words_pool).^2;
    [fval_pos, x_min_pos] = min(temp_pos);
    [fval_neg, x_min_neg] =  min(temp_neg);
    if (fval_neg < fval_pos)
        idx_min =  x_min_neg;
        y_min = -1;      
        min_values = temp_neg;
    else
        idx_min = x_min_pos; 
        y_min = 1;
        min_values = temp_pos;
    end    
end