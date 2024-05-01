%% Random example selection
% Teach positive vs. negative in Italian by randomly selecting the examples
% First, randomly selecting a Spanish word and translating it to Italian
% Second, directly selecting a Italian word uniformly at random
% 2023
%% Initialize Dataset
load("initialization.mat")
n_iterations = 100;
tau = 0.01;
%% To change
filename = 'random_it_results.mat'; 
%%
disp("Random from Spanish...")
acc_learner_random_estoit = zeros(n_learner, n_iterations+1);
e_learner_random_estoit = zeros(n_learner, n_iterations+1);
for ii = 1:n_learner
    disp(ii)
    learner = learner_init(:, ii);
    temp = (sign(x_test_it' * learner)-y_test_it)./2; 
    acc_learner_random_estoit(ii, 1) = 1 - sum(abs(temp))./ length(y_test_it);
    e_learner_random_estoit(ii, 1) = vecnorm(learner-theta_it);
    for jj = 1:n_iterations
        idx_min_learner = [];
        while(isempty(idx_min_learner))
            idx_random_estoit = randi([1, 1000], 1);
            temp2 = strcmp(words_it, dict_it{idx_random_estoit});
            idx_min_learner = find(temp2 == 1);
        end
        x_min = emb_norm_it(:, idx_min_learner);
        y_min = sign(theta' * emb_norm_es_commun(:, idx_random_estoit));
        
        learner = learner - ...
            tau*(learner'* x_min - y_min) * x_min; 
        temp = (sign(x_test_it' * learner)-y_test_it)./2; 
        acc_learner_random_estoit(ii, jj+1) = 1 - sum(abs(temp))./ length(y_test_it);
        e_learner_random_estoit(ii, jj+1) = vecnorm(learner-theta_it);
    end
end
accm_random_estoit = mean(acc_learner_random_estoit);
em_random_estoit = mean(e_learner_random_estoit);
%%
disp("Random in Italian...")
acc_learner_random = zeros(n_learner, n_iterations+1);
e_learner_random = zeros(n_learner, n_iterations+1);
len_random = length(x_test_it);
for ii = 1:n_learner
    disp(ii)
    learner = learner_init(:, ii);
    temp = (sign(x_test_it' * learner)-y_test_it)./2; 
    acc_learner_random(ii, 1) = 1 - sum(abs(temp))./ length(y_test_it);
    e_learner_random(ii, 1) = vecnorm(learner-theta_it);
    for jj = 1:n_iterations
        %idx_random = randi([1, len_random], 1);
        %y_min = y_test_it(idx_random); 
        %x_min = x_test_it(:, idx_random);
        idx_random = randi([1, 1e4], 1);
        x_min = emb_norm_it(:, idx_random);
        y_min = sign(theta_it' * x_min);
        
        learner = learner - ...
            tau*(learner'* x_min - y_min) * x_min; 
        temp = (sign(x_test_it' * learner)-y_test_it)./2; 
        acc_learner_random(ii, jj+1) = 1 - sum(abs(temp))./ length(y_test_it);
        e_learner_random(ii, jj+1) = vecnorm(learner-theta_it);
    end
end
accm_random = mean(acc_learner_random);
em_random = mean(e_learner_random);

save(filename, '-regexp', '^(?!emb_norm_it|emb_norm_es_commun|words_it|dict_it.*$).')
%% Plot 
% Accuracy
figure; plot(0:n_iterations, accm_random(1:n_iterations+1), 'g--', 'linewidth', 2)
hold on; plot(0:n_iterations, accm_random_estoit(1:n_iterations+1), '--', 'linewidth', 2)
grid on;
ylabel("Classification Accuracy")
xlabel("Iteration")
xlim([0, n_iterations])
legend("Random in learner's language", "Random from teacher's language", 'fontsize', 10)

% Plot Error
figure; semilogy(0:n_iterations, em_random(1:n_iterations+1), 'g', 'linewidth', 2)
hold on; semilogy(0:n_iterations, em_random_estoit(1:n_iterations+1), '--','linewidth', 2)
grid on;
%ylabel("Error Norm")
ylabel("$\|\mathbf{\theta} - \hat{\mathbf{\theta}}_i\|_2^2$", 'interpreter', 'latex')
xlabel("Iteration")
xlim([0, n_iterations])
legend("Random in learner's language", "Random from teacher's language",'fontsize', 10)