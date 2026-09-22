%% Structured order-5 controller from Salmon, section 3.4.1.
%
% K(s) = (b4*s^4 + b3*s^3 + b2*s^2 + b1*s) /
%        (s^5 + a4*s^4 + a3*s^3 + a2*s^2 + a1*s + a0)
%
% The missing constant numerator coefficient enforces K(0)=0. The controller
% denominator is parameterized by stable factors, and the nine free parameters
% are searched independently of any later nonlinear optimization.

clear; close all; clc;

this_dir = fileparts(mfilename('fullpath'));
rom_file = fullfile(this_dir, 'data_input', 'sysid_o16_d=3_ssest.mat');
output_file = fullfile(this_dir, 'data_input', ...
    'K_thesis_structured_order5.mat');
fallback_file = fullfile(this_dir, 'data_input', ...
    'K_thesis_structured_order5_fmincon.mat');
legacy_file = fullfile(this_dir, 'K_tuned_stable.mat');

rom_data = load(rom_file);
ROM = minreal(ss(rom_data.A, rom_data.B, rom_data.C, rom_data.D), ...
    1e-9, false);
assert(order(ROM) == 16 && size(ROM,1) == 1 && size(ROM,2) == 1);

design.minimum_decay = 0.03;
design.frequency = logspace(-4, 4, 2000);
design.n_random_starts = 80;
design.n_local_starts = 24;
design.seed = 4;
design.numerator_scale = [10, 2e4, 2e4, 3e4];

% Three stable real poles and one stable conjugate pair give a general enough
% fifth-order denominator while guaranteeing controller stability. The first
% five optimization variables are logarithms of positive quantities:
% [p1 p2 p3 omega zeta].
lower = [log(0.02)*ones(1,3), log(0.05), log(0.03), -20*ones(1,4)];
upper = [log(500)*ones(1,3), log(100), log(3), 20*ones(1,4)];

legacy = load(legacy_file);
[legacy_num, ~] = tfdata(tf(legacy.K_tuned), 'v');
legacy_poles = pole(legacy.K_tuned);
real_poles = sort(-real(legacy_poles(abs(imag(legacy_poles)) < 1e-8))).';
pair = legacy_poles(imag(legacy_poles) > 0);
omega0 = abs(pair(1));
zeta0 = -real(pair(1))/omega0;
theta0 = [log(real_poles), log(omega0), log(zeta0), ...
    legacy_num(1:4)./design.numerator_scale];
theta0 = min(max(theta0, lower), upper);

objective = @(theta) synthesis_cost(theta, ROM, design);
options = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'off', ...
    'MaxIterations', 500, 'MaxFunctionEvaluations', 6000, ...
    'OptimalityTolerance', 1e-7, 'StepTolerance', 1e-9);

% First rank broad deterministic samples cheaply, then refine only the best.
rng(design.seed);
starts = repmat(lower, design.n_random_starts, 1) + ...
    rand(design.n_random_starts, numel(lower)).* ...
    repmat(upper-lower, design.n_random_starts, 1);
starts(:,6:9) = 2*randn(design.n_random_starts,4);
nearby = theta0 + [0.8*randn(40,5), 0.8*randn(40,4)];
nearby = min(max(nearby, lower), upper);
starts = [theta0; nearby; starts];
start_cost = zeros(size(starts,1),1);
for index = 1:size(starts,1)
    start_cost(index) = objective(starts(index,:));
end
[~, ranking] = sort(start_cost);

best.theta = theta0;
best.cost = objective(theta0);
for index = 1:min(design.n_local_starts, numel(ranking))
    [theta, cost] = fmincon(objective, starts(ranking(index),:), ...
        [], [], [], [], lower, upper, [], options);
    if cost < best.cost
        best.theta = theta;
        best.cost = cost;
    end
end

[~, metrics, coefficients] = synthesis_cost(best.theta, ROM, design);
fprintf('Best search abscissa before acceptance: %.6e\n', ...
    metrics.closed_loop_abscissa);
method = 'fmincon fallback';
[A, B, C, D, K_tuned] = canonical_controller(coefficients);
fallback_metrics = metrics;
fallback_coefficients = coefficients;
fallback_method = method;
save(fallback_file, 'A', 'B', 'C', 'D', 'K_tuned', ...
    'fallback_coefficients', 'fallback_metrics', 'design', ...
    'fallback_method');

% When Robust Control Toolbox is available, refine the same exact transfer-
% function structure with systune. b0 remains fixed rather than merely being
% initialized to zero. The pole goal is soft (as in systune synthesis), while
% standalone controller stability is a hard constraint.
if license('test', 'Robust_Toolbox')
    [K_seed, ~] = make_controller(best.theta, design);
    [seed_num, seed_den] = tfdata(K_seed, 'v');
    K_block = tunableTF('K', 4, 5);
    K_block.Numerator.Value = seed_num(end-4:end);
    K_block.Numerator.Value(end) = 0;
    K_block.Numerator.Free = [true, true, true, true, false];
    K_block.Numerator.Scale = max(abs(K_block.Numerator.Value), 1);
    K_block.Denominator.Value = seed_den;
    K_block.Denominator.Scale = max(abs(seed_den), 1);

    tune_model = feedback(ROM*K_block, 1);
    closed_poles = TuningGoal.Poles(design.minimum_decay, 0, Inf);
    stable_controller = TuningGoal.ControllerPoles('K', 1e-4, 0, Inf);
    tune_options = systuneOptions('RandomStart', 20, 'Display', 'final');
    [tuned_model, f_soft, g_hard] = systune(tune_model, ...
        closed_poles, stable_controller, tune_options);
    K_systune = minreal(ss(getBlockValue(tuned_model, 'K')), 1e-9, false);
    systune_metrics = evaluate_controller(ROM, K_systune, design.frequency);
    if g_hard <= 1 && systune_metrics.closed_loop_abscissa < ...
            metrics.closed_loop_abscissa
        [num, den] = tfdata(tf(K_systune), 'v');
        num = num(end-4:end);
        num(end) = 0;
        leading = den(1);
        den = den/leading;
        num = num/leading;
        coefficients.a = fliplr(den(2:end));
        coefficients.b = fliplr(num(1:4));
        metrics = systune_metrics;
        metrics.f_soft = f_soft;
        metrics.g_hard = g_hard;
        method = 'systune';
    end
end

assert(metrics.controller_abscissa < 0, 'Controller is not stable.');
assert(metrics.closed_loop_abscissa <= -design.minimum_decay, ...
    'No structured controller achieved the required -0.03 decay.');

% Exact observable-canonical realization, equations (3.40)-(3.42).
[A, B, C, D, K_tuned] = canonical_controller(coefficients);
a0 = coefficients.a(1); a1 = coefficients.a(2);
a2 = coefficients.a(3); a3 = coefficients.a(4);
a4 = coefficients.a(5);
b1 = coefficients.b(1); b2 = coefficients.b(2);
b3 = coefficients.b(3); b4 = coefficients.b(4);

fprintf('\nStructured order-5 synthesis\n');
fprintf('method = %s\n', method);
fprintf('K(0) = %.3e\n', dcgain(K_tuned));
fprintf('controller abscissa = %.4e\n', metrics.controller_abscissa);
fprintf('closed-loop abscissa = %.4e (required <= -%.3f)\n', ...
    metrics.closed_loop_abscissa, design.minimum_decay);
fprintf('max|S| = %.3f; max|K*S| = %.3f\n', ...
    metrics.max_sensitivity, metrics.max_command_gain);
disp('denominator [1 a4 a3 a2 a1 a0]:');
disp([1, a4, a3, a2, a1, a0]);
disp('numerator [b4 b3 b2 b1 0]:');
disp([b4, b3, b2, b1, 0]);

save(output_file, 'A', 'B', 'C', 'D', 'K_tuned', 'coefficients', ...
    'metrics', 'design', 'method');

function [cost, metrics, coefficients] = synthesis_cost(theta, plant, design)
    [K, coefficients] = make_controller(theta, design);
    closed = feedback(plant, K);
    metrics.controller_abscissa = max(real(pole(K)));
    metrics.closed_loop_abscissa = max(real(pole(closed)));

    % Primary goal: minimize the rightmost closed-loop pole. Stability of K
    % is guaranteed by its factorized denominator parameterization.
    cost = metrics.closed_loop_abscissa + 1e-9*sum(theta(6:9).^2);

    if nargout > 1
        metrics = evaluate_controller(plant, K, design.frequency);
    end
end

function metrics = evaluate_controller(plant, K, w)
    metrics.controller_abscissa = max(real(pole(K)));
    metrics.closed_loop_abscissa = max(real(pole(feedback(plant, K))));
    loop = reshape(freqresp(plant*K, w), 1, []);
    S = 1./(1 + loop);
    K_w = reshape(freqresp(K, w), 1, []);
    metrics.max_sensitivity = max(abs(S));
    metrics.max_command_gain = max(abs(K_w.*S));
end

function [K, coefficients] = make_controller(theta, design)
    real_rates = exp(theta(1:3));
    omega = exp(theta(4));
    zeta = exp(theta(5));
    denominator = conv(poly(-real_rates), [1, 2*zeta*omega, omega^2]);
    numerator = [theta(6:9).*design.numerator_scale, 0];
    K = tf(numerator, denominator);
    coefficients.a = fliplr(denominator(2:end)); % [a0 ... a4]
    coefficients.b = fliplr(numerator(1:end-1)); % [b1 ... b4]
end

function [A, B, C, D, K] = canonical_controller(coefficients)
    a0 = coefficients.a(1); a1 = coefficients.a(2);
    a2 = coefficients.a(3); a3 = coefficients.a(4);
    a4 = coefficients.a(5);
    b1 = coefficients.b(1); b2 = coefficients.b(2);
    b3 = coefficients.b(3); b4 = coefficients.b(4);
    A = [0, 0, 0, 0, -a0;
         1, 0, 0, 0, -a1;
         0, 1, 0, 0, -a2;
         0, 0, 1, 0, -a3;
         0, 0, 0, 1, -a4];
    B = [0; b1; b2; b3; b4];
    C = [0, 0, 0, 0, 1];
    D = 0;
    K = minreal(ss(A,B,C,D), 1e-9, false);
end
