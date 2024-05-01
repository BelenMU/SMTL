%% Machine Teaching for a Synthetic Dataset
% 2023
% Analysis of impact in performance caused by deviations of examples from
% perfect orthonormality.

%{ 
OUTPUTS
- acc_X: Test accuracy, i.e., percentage of correct classification of
examples in the testing set as iterations progress. 
- accm_X: Test accuracy averaged over all the initializations on the
learner's state.
- C_mt_X: Covariance matrix of teacher's posterior about learner's state
- e_mt_X: MSE between the learner's state and the ground truth across
iterations.
- e_estimation_X: MSE between the learner's state and the teacher's
estimation about the learner's state accross iterations.
- e_mean_X: MSE between the ground truth and the teacher's
estimation about the learner's state accross iterations.
- em_mt_X: MSE averaged over all the initializations on the
learner's state.
- x_mt_X: Examples generated by teacher and shown to learner.
- y_mt_X: Label generated by teacher, corresponding to x_mt_X.
%}

%% Initialize 
var_ortho = 90; % Variance of the noise by which the examples will deviate from perfect orthonormality (In degrees)
filename = "syn2_nonorthonormalmapping_results.mat";
n_learner = 50; % Number of initialization of learners to be examined
n_iterations = 400; % Number of iterations
tau = 0.01; % Learning Rate
X_min = -2.5; % Upper bound for coordinates of the examples generated by teacher
X_max = 2.5; % Lower bound for coordinates of the examples generated by teacher
sig = 0.1; % Measurement Noise, change to get the corresponding lines in the plot

t = rand(1) * 2*pi; % Randomly chosen rotation angle
Rx = [1 0 0; 0 cos(t) -sin(t); 0 sin(t) cos(t)]; % Rotation matrix
z = sqrt(var_ortho * pi/180); % Typical deviation of noise modeling non-orthonormal imperfections

% Sample learner initial state
learner_init =  rand(n_learner, 3).*2-1;
% Empirical covariance of uniformly distributed random vectors with
% coordinates bounded between -1 and 1.
C_init = [0.3276   -0.0005   -0.0154 
   -0.0005    0.3464   -0.0070
   -0.0154   -0.0070    0.3287];


% Replicate dataset based on synth2 (Yang and Loog, 2018)
n_test = 2000; % Number of points on bigger cloud
n_testab = 250; % Number of points on smaller clouds
mu_1 = [0, -1];
mu_2 = [0, 1];
mu_1a = [2, -2];
mu_1b = [2, 2];
mu_2a = [-2 -2];
mu_2b = [-2 2];
% Example points
x1_test = randn(n_test, 2)./3 + mu_1;
x1_test = [x1_test; randn(n_testab, 2)./3 + mu_1a];
x1_test = [x1_test; randn(n_testab, 2)./3 + mu_1b];
x2_test = randn(n_test, 2)./3 + mu_2;
x2_test = [x2_test; randn(n_testab, 2)./3 + mu_2a];
x2_test = [x2_test; randn(n_testab, 2)./3 + mu_2b];
x_test = [ones(2*n_test+ 4*n_testab, 1), [x1_test; x2_test]];
% Labels
y_test = [ones(n_test+2*n_testab, 1); -1.*ones(n_test+2*n_testab, 1)];

% Ground truth in learner's space is the linear least-square solution
theta_learner = lsqlin([ones(2*n_test+ 4*n_testab, 1) ,[ x1_test; x2_test]], ...
    [ones(n_test+ 2*n_testab, 1); -1.*ones(n_test+ 2*n_testab, 1)]);

% Ground truth in equivalent rotated teacher's space
theta_teacher = Rx' * theta_learner; % Transpose inverts rotation matrix

%% Machine Teaching No Feedback - Synthetic
disp("No Feedback - Synthetic...")
acc_mt_no_syn = zeros(n_learner, n_iterations+1);
e_mt_no_syn = zeros(n_learner, n_iterations+1);
e_estimation_no_syn = zeros(n_learner, n_iterations+1);
e_mean_no_syn = zeros(n_learner, n_iterations+1);
x_mt_no_syn = zeros(3, n_learner, n_iterations);
y_mt_no_syn = zeros(n_learner, n_iterations);
C_mt_no_syn = zeros(n_learner, n_iterations+1);
learner_no_syn = zeros(3, n_learner, n_iterations+1);
for ii = 1:n_learner
    learner = learner_init(ii, :)';
    % Compute starting test-accuracy, MSE and decomposition, Cov, mean...
    learner_no_syn(:, ii, 1) = learner;
    temp = (sign(x_test * learner)-y_test)./2;
    acc_mt_no_syn(ii, 1) = 1 - sum(abs(temp))./ length(y_test);
    C = C_init;
    C_mt_no_syn(ii, 1) = norm(C);
    mu = zeros(3, 1); 
    e_mt_no_syn(ii, 1) = vecnorm(learner-theta_learner);
    e_estimation_no_syn(ii, 1) = vecnorm(learner-Rx*mu);
    e_mean_no_syn(ii, 1) = vecnorm(mu-theta_teacher);
    for jj = 1:n_iterations
        % Select example that max T
        [x_min, y_min] = maximizer_MT(C, mu, theta_teacher,tau, X_min, X_max);
        x_mt_no_syn(:,ii, jj) = x_min;
        y_mt_no_syn(ii, jj) = y_min;
        % Update estimations about learner according to Algorithm 1
        mu = mu - ...
            tau*(mu'*x_min - y_min) * x_min; 
        % Rotate example for learner and add some noise to model
        % imperfection
        t_learner = t + randn(1)*z;
        Rx = [1 0 0; 0 cos(t_learner) -sin(t_learner); 0 sin(t_learner) cos(t_learner)]; % Noisy rotation matrix
        x_learner = Rx * x_min;
        learner = learner - ...
            tau*(learner'*x_learner - y_min) * x_learner; 
        learner_no_syn(:, ii, jj + 1) = learner;
        % Compute performance metrics
        temp = (sign(x_test * learner)-y_test)./2; 
        acc_mt_no_syn(ii, jj+1) = 1 - sum(abs(temp))./ length(y_test);
        e_mt_no_syn(ii, jj+1) = vecnorm(learner-theta_learner);
        e_estimation_no_syn(ii, jj+1) = vecnorm(learner-Rx*mu);
        e_mean_no_syn(ii, jj+1) = vecnorm(mu-theta_teacher);
        C = C - tau*C*(x_min*x_min') - tau*(x_min*x_min')*C +...
            tau^2*x_min'*C*x_min*(x_min*x_min');    
        C_mt_no_syn(ii, jj+1) = norm(C);
    end
end
accm_mt_no_syn = mean(acc_mt_no_syn);
em_mt_no_syn = mean(e_mt_no_syn);
em_estimation_no_syn = mean(e_estimation_no_syn);
em_mean_no_syn = mean(e_mean_no_syn);
Cm_mt_no_syn = mean(C_mt_no_syn);

%% Machine Teaching Noisy Feedback - Synthetic
disp("Noisy Feedback...")

acc_mt_kalman_syn  = zeros(n_learner, n_iterations+1);
e_mt_kalman_syn  = zeros(n_learner, n_iterations+1);
e_estimation_kalman_syn  = zeros(n_learner, n_iterations+1);
e_mean_kalman_syn  = zeros(n_learner, n_iterations+1);
C_mt_kalman_syn  = zeros(n_learner, n_iterations+1);

for ii = 1:n_learner
    learner = learner_init(ii, :)';
    mu = zeros(3, 1);
    temp = (sign(x_test * learner)-y_test)./2; %24D_normalized
    acc_mt_kalman_syn (ii, 1) = 1 - sum(abs(temp))./ length(y_test); 
    e_mt_kalman_syn (ii, 1) = vecnorm(learner-theta_learner);
    e_estimation_kalman_syn(ii, 1) = vecnorm(learner-Rx*mu);
    e_mean_kalman_syn(ii, 1) = vecnorm(mu-theta_teacher);
    C = C_init;
    C_mt_kalman_syn (ii, 1) = norm(C);
    for jj = 1:n_iterations
        [x_min, y_min] = maximizer_MT(C, mu, theta_teacher,tau, X_min, X_max);
        % Rotate example for learner and add some noise to model
        % imperfection
        t_learner = t + randn(1)*z;
        Rx = [1 0 0; 0 cos(t_learner) -sin(t_learner); 0 sin(t_learner) cos(t_learner)]; % Noisy rotation matrix
        x_learner = Rx * x_min;
        learner = learner - ...
            tau*(learner'*x_learner - y_min) * x_learner; 
        learner_no_syn(:, ii, jj + 1) = learner;
        % Compute Feedback
        s = learner'*x_learner + randn(1)*sqrt(sig);
        % Update Teacher's Estimation of Learner   
        % 1) Extrapolation
        F = eye(3) - tau * (x_min*x_min');
        mu_prev = F*mu + tau * y_min * x_min; 
        C_prev = F*C*F'; 
        K = C_prev * x_min / (x_min'*C_prev*x_min + sig);   
        % 2) Update
        C = (eye(3)-K*x_min') * C_prev * (eye(3)-K*x_min')' + sig*(K*K');
        mu = mu_prev + K*(s - x_min'*mu_prev);
        % Compute Accuracy        
        temp = (sign(x_test * learner)-y_test)./2; %24D_normalized
        acc_mt_kalman_syn (ii, jj+1) = 1 - sum(abs(temp))./ length(y_test);
        e_mt_kalman_syn (ii, jj+1) = vecnorm(learner-theta_learner);
        e_estimation_kalman_syn(ii, jj+1) = vecnorm(learner-Rx*mu);
        e_mean_kalman_syn(ii, jj+1) = vecnorm(mu-theta_teacher);
        C_mt_kalman_syn (ii, jj + 1) = norm(C);
    end
end
accm_mt_kalman_syn  = mean(acc_mt_kalman_syn );
em_mt_kalman_syn  = mean(e_mt_kalman_syn );
em_estimation_kalman_syn = mean(e_estimation_kalman_syn);
em_mean_kalman_syn = mean(e_mean_kalman_syn);
Cm_mt_kalman_syn = mean(C_mt_kalman_syn);
save(filename)
disp("Done")

%% Plot Figures
% Plot Accuracy
figure; %semilogy(0:n_iterations, accm_mt_omni_syn(1:n_iterations+1), 'k',  'linewidth', 2)
hold on; semilogy(0:n_iterations, accm_mt_kalman_syn(1:n_iterations+1), 'c--', 'linewidth', 2)
hold on; semilogy(0:n_iterations, accm_mt_no_syn(1:n_iterations+1), 'b', 'linewidth', 2)
legend( "Algorithm 2, \sigma^2 = 0.1",...
    "Algorithm 1", 'NumColumns',2, 'Fontsize', 10, 'location', 'sw')
grid on;
ylabel("Classification Accuracy")
xlabel("Iteration")
xlim([0, n_iterations])

% Plot Error
figure; %semilogy(0:n_iterations, em_mt_omni_syn(1:n_iterations+1), 'k',  'linewidth', 2)
hold on; semilogy(0:n_iterations, em_mt_kalman_syn(1:n_iterations+1), 'c--', 'linewidth', 2)
hold on; semilogy(0:n_iterations, em_mt_no_syn(1:n_iterations+1), 'b', 'linewidth', 2)
legend("Algorithm 2, \sigma^2 = 0.1",...
    "Algorithm 1",'NumColumns',2, 'Fontsize', 10, 'location', 'sw')
grid on;
ylabel("$\|\widehat{\mathbf{\theta}}_i- {\mathbf{\theta}}\|_2$", "interpreter", "latex")
xlabel("Iteration")
xlim([0, 90])

% Plot Covariance
figure; semilogy(0:n_iterations, Cm_mt_kalman_syn, 'r', 'linewidth', 2)
hold on; plot(0:n_iterations, Cm_mt_no_syn, 'b', 'linewidth', 2)
legend("Algorithm 2, \sigma^2 = 0.1", "Algorithm 1",'Fontsize', 11)
grid on;
ylabel("$\|\mathbf{C}_i\|_2$", "interpreter", "latex",'Fontsize', 12)
xlabel("$i$", "interpreter", "latex",'Fontsize', 12)
xlim([0, n_iterations])

% Plot Error Source
figure; semilogy(0:n_iterations, em_mt_kalman_syn, 'linewidth', 2.5)
hold on; semilogy(0:n_iterations, em_estimation_kalman_syn,':', 'linewidth', 2.5)
hold on; semilogy(0:n_iterations, em_mean_kalman_syn, '--', 'linewidth', 2.5)
grid on
legend("$\|\mathbf{\theta} - \hat{\mathbf{\theta}}_i\|_2$",...
"$\|\mathbf{\theta} - \hat{\mathbf{\theta}}_i\|_2$ ", "$\|\mathbf{\theta} - \mathbf{\mu}_i\|_2$", 'interpreter', 'latex', 'fontsize', 11)
xlabel("$i$", "interpreter", "latex")
xlim([0, n_iterations])