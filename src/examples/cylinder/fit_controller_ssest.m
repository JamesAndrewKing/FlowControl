% filepath: fit_controller_ssest.m
% --- Load frequency response data ---
data_G = load('data_output/nyquist/G_vals.mat');
data_omega = load('data_output/nyquist/omega_range.mat');
omega_data = data_omega.omega_range(:).';
G_vals_data = data_G.G_vals(:).';

% --- Define desired frequency grid ---
omega_range = logspace(-2, 2, 100); % 10^-2 to 10^2 rad/s

% --- Interpolate data to desired omega_range if needed ---
G_vals = interp1(omega_data, G_vals_data, omega_range, 'linear', 'extrap');

%% --- Plot Bode plot directly from frequency response data ---
figure;
% Calculate magnitude in dB
mag_dB = 20*log10(abs(G_vals));
% Calculate phase in degrees
phase_deg = rad2deg(unwrap(angle(G_vals)));

% Top plot: Magnitude
subplot(2,1,1);
semilogx(omega_range, mag_dB);
grid on;
title('Bode Plot from Raw Frequency Data');
ylabel('Magnitude (dB)');

% Bottom plot: Phase
subplot(2,1,2);
semilogx(omega_range, phase_deg);
grid on;
xlabel('Frequency (rad/s)');
ylabel('Phase (deg)');
% --- End of Bode plot section ---

%%

% --- Fit a reduced-order model (order 16 as in the text) ---
data_id = idfrd(G_vals, omega_range, 0); % SISO frequency response data
order_ROM = 16;
ROM = ssest(data_id, order_ROM, 'Ts', 0);

% --- Define a tunable controller in observable canonical form (order 5) ---
order_K = 5;
a = realp('a', zeros(order_K, 1));
b = realp('b', zeros(order_K, 1));

% Enforce K(0)=0 by fixing b0=0
b.Value(1) = 0;
b.Free(1) = false;

% State-space matrices for observable canonical form
% State-space matrices for observable canonical form
L = diag(ones(order_K-1, 1), -1); % Subdiagonal of ones
L(:, end) = -a; % Last column is -a
M = b;
N = [zeros(1, order_K-1), 1];

% Create the tunable state-space controller
K_ss = ss(L, M, N, D);
K_ss.StateName = repmat({'x'}, order_K, 1);
K_ss.Name = 'K_ss';

% --- Closed-loop interconnection ---
% Use AnalysisPoint for systune to identify the loop
CL_io = AnalysisPoint('CL_io');
CL = feedback(series(K_ss, ROM), CL_io);

% --- Tuning goals ---
% All closed-loop poles should have real part < -0.03
% This means the decay rate should be > 0.03
Req1 = TuningGoal.Poles(1, 0.03, Inf); % MaxFreq=Inf to apply to all poles

% --- Structured synthesis ---
% Use systune with the closed-loop model and the tuning goal
[CL_tuned, fSoft] = systune(CL, Req1);

% --- Extract tuned controller and check results ---
if fSoft < 1
    disp('Tuning successful.');
    % Extract the tuned controller block
    K_final = getBlockValue(CL_tuned, 'K_ss');

    % Display the final controller
    disp('Tuned controller K_final:');
    disp(K_final);

    % Verify closed-loop poles
    CL_final = feedback(series(K_final, ROM), 1);
    disp('Final closed-loop poles:');
    disp(pole(CL_final));

    % --- Bode comparison ---
    figure;
    bodeplot(ROM, K_final, omega_range);
    legend('ROM', 'Controller');
    title('Bode Plot: ROM vs. Tuned Controller');
    grid on;

    % --- Save controller ---
    % save('K_structured.mat', 'K_final');
else
    disp('Tuning did not meet the requirements (fSoft >= 1).');
end