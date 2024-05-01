%% Machine Teaching for a Synthetic Dataset
% 2023
% Analysis of empirical performance of baseline Machine Teaching algorithms 
% described on "Iterative Machine Teaching", namely in the section titled
% "Imitation Teaching". It includes the probind as well as the teaching
% phase.
%%
sig = 1e-3; % To change to get the corresponding lines in the plot
filename = "syn2_results_learningforomniscience.mat";
% Load previous, to have the same initializations, including randomly sampled
% learner's starting state
load("syn2_results.mat") 
%% Learning for Omniscience
delta = 1e-3;
d = 2; % Dimension of examples without the first coordinate =1
e_mt_kalman_delta1e3  = zeros(n_learner, n_iterations+1);
e_estimation_kalman_delta1e3  = zeros(n_learner, n_iterations+1);
e_mean_kalman_delta1e3  = zeros(n_learner, n_iterations+1);
C_mt_kalman_delta1e3  = zeros(n_learner, n_iterations+1);
for ii = 1:n_learner
    disp(ii)
    learner = learner_init(ii, :)';
    mu = zeros(3, 1);
    e_mt_kalman_delta1e3 (ii, 1) = vecnorm(learner-theta);
    e_estimation_kalman_delta1e3(ii, 1) = vecnorm(learner-mu);
    e_mean_kalman_delta1e3(ii, 1) = vecnorm(mu-theta);
    C = C_init;
    C_mt_kalman_delta1e3 (ii, 1) = norm(C);
    nC = norm(C);
    for jj = 1:n_iterations
        if nC > delta % Phase 1: Probing
            % Select example that minimizes uncertainty
            [x_min, ~] = probing_LfO( C, mu, X_min, X_max, sig);
            % Compute Feedback
            s = learner'*x_min + randn(1)*sqrt(sig);
            % Update Teacher's Estimation of Learner according to feedback  
            % 1)Extrapolation
            K = C * x_min / (x_min'*C*x_min + sig);   
            % 2) Update
            C = (eye(d+1) - K*x_min') *C;%* (eye(d+1) - K*x_min')' + sig*(K*K');
            mu = mu + K*(s - x_min'*mu);
            nC = norm(C);
        
        else % Phase 2: Teaching
            % Select example maximizing T as if teacher were omniscient
            [x_min, y_min] = maximizer_MT(zeros(3, 3), mu, theta,tau, X_min, X_max);
            % Update learner
            learner = learner - ...
                tau*(learner'*x_min - y_min) * x_min;    
            % Update Teacher's Estimation of Learner according to dynamical
            % moel
            mu = mu - tau*(mu'*x_min - y_min) * x_min;             
        end
       
        % Compute Accuracy        
        e_mt_kalman_delta1e3 (ii, jj+1) = vecnorm(learner-theta);
        e_estimation_kalman_delta1e3(ii, jj+1) = vecnorm(learner-mu);
        e_mean_kalman_delta1e3(ii, jj+1) = vecnorm(mu-theta);
        C_mt_kalman_delta1e3 (ii, jj + 1) = nC;
    end
end
em_mt_kalman_delta1e3  = mean(e_mt_kalman_delta1e3 );
em_estimation_kalman_delta1e3 = mean(e_estimation_kalman_delta1e3); %sqrt(mean(e_estimation_kalman_delta1e3.^2));
em_mean_kalman_delta1e3 = mean(e_mean_kalman_delta1e3);
Cm_mt_kalman_delta1e3 = mean(C_mt_kalman_delta1e3);
save(filename)
%% Plot Covariance
figure; semilogy(0: n_iterations,Cm_mt_kalman_syn, 'r','linewidth', 2)
hold on; semilogy(0: n_iterations, Cm_mt_no_syn, 'b', 'linewidth', 2)
hold on; semilogy(0: n_iterations, Cm_mt_kalman_delta1e3,'--','linewidth', 2)
legend("Noisy Feedback","No Feedback",...%
    "LfO, \delta = 10^{-3}",...
     "fontsize", 10)
grid on;
ylabel("$\|\mathbf{C}_i\|_2$", "interpreter", "latex",'Fontsize', 12)
xlabel("$i$", "interpreter", "latex",'Fontsize', 12)
%% Plot Error
figure; semilogy(0: n_iterations, em_mt_kalman_syn, 'r', 'linewidth', 2)
hold on; plot(0:n_iterations,em_mt_no_syn, 'b', 'linewidth', 2)
hold on; plot(0:n_iterations,em_mt_kalman_delta1e3, '--', 'linewidth', 2)
legend("Noisy Feedback","No Feedback",...
    "LfO, \delta = 10^{-3}",...
     "fontsize", 10)
grid on;
ylabel("$\|\widehat{\mathbf{\theta}}_i- {\mathbf{\theta}}\|_2$", "interpreter", "latex")
xlabel("$i$", "interpreter", "latex",'Fontsize', 12)
xlim([0, n_iterations+1])