%% Predefined Pool es -> it
% Teach positive vs. negative from Spanish to Italian using only the 10000
% most used Spanish words
% 2023
%% Initialize Dataset
load('initialization.mat')
filename = 'predefinedPoolMT_results.mat'; 
n_iterations = 100; % Set to 3000 to rederive Fig. 8
%{
d = 300;
n_learner = 200;

learner_init = rand(d, 250).*2-1;
learner_init = learner_init ./ vecnorm(learner_init);
C_init = cov(learner_init');
learner_init = learner_init(:, 1:n_learner);
%n_learner = 25;

tau = 0.01;
sig = 0.0129;
n_iterations = 1000;
%}
%% No Feedback 
disp("No Feedback - Predefined Pool ...")
% Initialize metrics
acc_teach_no = zeros(n_learner, n_iterations+1);
e_model_no = zeros(n_learner, n_iterations+1);
y_no = zeros(n_learner, n_iterations);
C_no = zeros(n_learner, n_iterations+1);
k_no = zeros(n_learner, n_iterations);
idx_no = zeros(n_learner, n_iterations);
idx_learner_it_no = zeros(n_learner, n_iterations);
acc_learner_it_no = zeros(n_learner, n_iterations+1);
e_learner_it_no = zeros(n_learner, n_iterations+1);

for ii =1:n_learner
    disp(ii)
    mu = zeros(d, 1); 
    learner = learner_init(:, ii);
    e_model_no(ii, 1) = vecnorm(mu-theta);
    temp = (sign(x_train' * mu)-y_train)./2; 
    acc_teach_no(ii, 1) = 1 - sum(abs(temp))./ length(y_train);
    C = C_init;
    C_no(ii, 1) = norm(C);
    temp = (sign(x_test_it' * learner)-y_test_it)./2; 
    acc_learner_it_no(ii, 1) = 1 - sum(abs(temp))./ length(y_test_it);
    e_learner_it_no(ii, 1) = vecnorm(learner-theta_it);
   
    for jj = 1:n_iterations
        [idx, y_min, min_values] = maximizer_predefinedPool( C, mu, theta, tau, emb_norm_es_commun);
        y_no(ii, jj) = y_min;
        
        % Transalte example to learner's space
        temp2 = strcmp(words_it, dict_it{idx});
        idx_min_learner = find(temp2 == 1);
        if(isempty(idx_min_learner))
            %disp('not found')
            %disp(idx)
            aux = 2;
            while(isempty(idx_min_learner))
                [~, idx] = mink(min_values, aux);
                temp2 = strcmp(words_it, dict_it{idx(end)});
                idx_min_learner = find(temp2 == 1);
                aux = aux + 1;
            end          
            %disp('substituting for')
            idx = idx(end);
            %disp(idx)            
        end
        
        idx_no(ii, jj) = idx;
        x_min = emb_norm_es_commun(:, idx);
        
        idx_learner_it_no(ii, jj) = idx_min_learner;
        x_min_learner = emb_norm_it(:, idx_min_learner);
        
        % Update learner        
        learner = learner - ...
            tau*(learner'*x_min_learner - y_min) * x_min_learner; 
        % Update Teacher's Estimation of Learner  
        mu = mu - ...
            tau*(mu'*x_min - y_min) * x_min; 
        C = C - tau*C*(x_min*x_min') - tau*(x_min*x_min')*C +...
            tau^2*x_min'*C*x_min*(x_min*x_min');  
        
        % Track performance
        temp = (sign(x_train' * mu)-y_train)./2; 
        acc_teach_no(ii, jj+1) = 1 - sum(abs(temp))./ length(y_train);
        e_model_no(ii, jj+1) = vecnorm(mu-theta);  
        C_no(ii, jj+1) = norm(C); 
        temp = (sign(x_test_it' * learner)-y_test_it)./2; 
        acc_learner_it_no(ii, jj+1) = 1 - sum(abs(temp))./ length(y_test_it);
        e_learner_it_no(ii, jj+1) = vecnorm(learner-theta_it);
    end
end
accm_teach_no = mean(acc_teach_no);
em_model_no = mean(e_model_no);
accm_learner_it_no = mean(acc_learner_it_no);
em_learner_it_no = mean(e_learner_it_no);
Cm_no = mean(C_no);

save(filename, '-regexp', '^(?!dict_it|words_es_commun|emb_norm_it|theta|theta_it|words_es_commun|words_it|x_train|x_test_it|y_train|y_test_it|temp.*$).')
%%
% Noisy Feedback 
disp("Noisy Feedback - Rescalable Pool ...")
acc_teach_noisy = zeros(n_learner, n_iterations+1);
e_model_noisy = zeros(n_learner, n_iterations+1);
y_noisy = zeros(n_learner, n_iterations);
C_noisy = zeros(n_learner, n_iterations+1);
k_noisy = zeros(n_learner, n_iterations);
idx_noisy = zeros(n_learner, n_iterations);
idx_learner_it_noisy = zeros(n_learner, n_iterations);
acc_learner_it_noisy = zeros(n_learner, n_iterations+1);
e_learner_it_noisy = zeros(n_learner, n_iterations+1);

for ii =1:n_learner
    disp(ii)
    mu = zeros(d, 1); 
    learner = learner_init(:, ii);
    e_model_noisy(ii, 1) = vecnorm(mu-theta);
    temp = (sign(x_train' * mu)-y_train)./2; 
    acc_teach_noisy(ii, 1) = 1 - sum(abs(temp))./ length(y_train);
    C = C_init;
    C_noisy(ii, 1) = norm(C);
    temp = (sign(x_test_it' * learner)-y_test_it)./2; 
    acc_learner_it_noisy(ii, 1) = 1 - sum(abs(temp))./ length(y_test_it);
    e_learner_it_noisy(ii, 1) = vecnorm(learner-theta_it);
   
    for jj = 1:n_iterations
        [idx, y_min, min_values] = maximizer_predefinedPool( C, mu, theta, tau, emb_norm_es_commun);
        y_noisy(ii, jj) = y_min;
        
        % Transalte example to learner's space
        temp2 = strcmp(words_it, dict_it{idx});
        idx_min_learner = find(temp2 == 1);
        if(isempty(idx_min_learner))
            %disp('not found')
            %disp(idx)
            aux = 2;
            while(isempty(idx_min_learner))
                [~, idx] = mink(min_values, aux);
                temp2 = strcmp(words_it, dict_it{idx(end)});
                idx_min_learner = find(temp2 == 1);
                aux = aux + 1;
            end          
            %disp('substituting for')
            idx = idx(end);
            %disp(idx)            
        end
        
        idx_noisy(ii, jj) = idx;
        x_min = emb_norm_es_commun(:, idx);
        
        idx_learner_it_noisy(ii, jj) = idx_min_learner;
        x_min_learner = emb_norm_it(:, idx_min_learner);
        
        % Update learner        
        learner = learner - ...
            tau*(learner'*x_min_learner - y_min) * x_min_learner; 
        % Send feedback
        s = learner'*x_min_learner;
        
        % Update Teacher's Estimation of Learner   
        % Extrapolation
        F = eye(d) - tau * (x_min*x_min');
        mu_prev = F*mu + tau * y_min * x_min; 
        C_prev = F*C*F'; 
        K = C_prev * x_min / (x_min'*C_prev*x_min + sig);   
        % Update
        C = (eye(d)-K*x_min') * C_prev * (eye(d)-K*x_min')' + sig*(K*K');
        mu = mu_prev + K*(s - x_min'*mu_prev);

        % Track performance
        temp = (sign(x_train' * mu)-y_train)./2; 
        acc_teach_noisy(ii, jj+1) = 1 - sum(abs(temp))./ length(y_train);
        e_model_noisy(ii, jj+1) = vecnorm(mu-theta);  
        C_noisy(ii, jj+1) = norm(C); 
        temp = (sign(x_test_it' * learner)-y_test_it)./2; 
        acc_learner_it_noisy(ii, jj+1) = 1 - sum(abs(temp))./ length(y_test_it);
        e_learner_it_noisy(ii, jj+1) = vecnorm(learner-theta_it);
    end
    save(filename, '-regexp', '^(?!dict_it|words_es_commun|emb_norm_it|theta|theta_it|words_es_commun|words_it|x_train|x_test_it|y_train|y_test_it|temp.*$).')
end
accm_teach_noisy = mean(acc_teach_noisy);
em_model_noisy = mean(e_model_noisy);
accm_learner_it_noisy = mean(acc_learner_it_noisy);
em_learner_it_noisy = mean(e_learner_it_noisy);
Cm_noisy = mean(C_noisy);
save(filename, '-regexp', '^(?!dict_it|words_es_commun|emb_norm_it|theta|theta_it|words_es_commun|words_it|x_train|x_test_it|y_train|y_test_it|temp.*$).')
%% Plot Accuracy
figure;  %plot(0:n_iterations, accm_teach_noisy(1:n_iterations+1), 'b', 'linewidth', 2); hold on;  
plot(0:n_iterations, accm_learner_it_no(1:n_iterations+1), 'k', 'linewidth', 2)
hold on; plot(0:n_iterations, accm_learner_it_noisy(1:n_iterations+1), 'r', 'linewidth', 2)
%hold on;  plot(0:n_iterations, accm_random(1:n_iterations+1), 'g', 'linewidth', 2)
legend("No feedback", "Noisy feedback, \sigma^2 = 0.0129", 'fontsize', 11 )
grid on;
ylabel("Classification Accuracy")
xlabel("Iteration")
xlim([0, n_iterations])
%% Plot Error
figure;  semilogy(0:n_iterations, em_learner_it_no(1:n_iterations+1), 'k', 'linewidth', 2)
hold on;  semilogy(0:n_iterations, em_learner_it_noisy(1:n_iterations+1), 'r', 'linewidth', 2)
grid on;
ylabel("Error Norm")
legend("No feedback", "Noisy feedback, \sigma^2 = 0.0129", 'fontsize', 11 )
xlabel("Iteration")
xlim([0, n_iterations])
%{
%% Plot Covariance
figure;semilogy(0:n_iterations, Cm_no, 'k', 'linewidth', 2)
hold on;semilogy(0:n_iterations, Cm_noisy, 'r', 'linewidth', 2)
ylabel("$\|\mathbf{C}_i\|_2$", "interpreter", "latex",'Fontsize', 12)
xlabel("$i$", "interpreter", "latex",'Fontsize', 12)
legend("No feedback", "Noisy feedback, \sigma^2 = 0.0129", 'fontsize', 11 )
xlim([0, n_iterations])
grid on
%}