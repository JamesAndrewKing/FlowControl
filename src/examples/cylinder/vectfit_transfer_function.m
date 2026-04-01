% --- Load Data ---
data_G = load('data_output/nyquist/G_vals.mat');
data_omega = load('data_output/nyquist/omega_range.mat');
G_vals = data_G.G_vals(:).';
omega_range = data_omega.omega_range(:).';

freq_Hz = omega_range / (2*pi);
s = 1j * omega_range;                  % Row vector

% --- Initial Poles for VECTFIT (increase as needed) ---
n_poles_real = 3;
n_poles_cmplx = 2;
init_poles = -logspace(log10(min(omega_range(omega_range>0))), log10(max(omega_range)), n_poles_real);
cplx_freqs = logspace(log10(min(omega_range(omega_range>0))), log10(max(omega_range)), n_poles_cmplx);
for k = 1:n_poles_cmplx
    init_poles = [init_poles, -cplx_freqs(k) + 1j*cplx_freqs(k), -cplx_freqs(k) - 1j*cplx_freqs(k)];
end

weight = ones(1, length(s));           % No weighting

% --- VECTFIT Options ---
opts = struct();
opts.relax = 1;
opts.stable = 1;
opts.asymp = 3; % Fit D and E
opts.spy1 = 0;
opts.spy2 = 0;
opts.logx = 1;
opts.logy = 1;
opts.errplot = 0;
opts.phaseplot = 0;
opts.legend = 0;

% --- Run VECTFIT ---
[SER, poles, rmserr, fit, opts] = vectfit3(G_vals, s, init_poles, weight, opts);

% --- Extract residues, D, E ---
residues = SER.C;  % For SISO, this is a row vector
d = SER.D;         % Constant term
h = SER.E;         % Proportional term

% --- Build Transfer Function ---
G = 0;
for k = 1:length(poles)
    G = G + residues(k) * tf(1, [1 -poles(k)]);
end
G = G + d + h*tf([1 0], 1);
G = minreal(G);
G = force_real_tf(G);

disp('Fitted transfer function:');
G

% --- Plot Fit Quality ---
figure;
plot(freq_Hz, abs(G_vals), 'b', 'DisplayName', 'Original');
hold on;
[mag_fit, ~] = bode(G, omega_range);
plot(freq_Hz, squeeze(mag_fit), 'r--', 'DisplayName', 'Fitted');
xlabel('Frequency [Hz]');
ylabel('|G(j\omega)|');
legend('Original', 'Fitted');
title('Fit Quality');
grid on;

% --- Weighting Function for Loop Shaping ---
kW = 1.0; % Tune as needed
aW = 2.0; % Tune as needed
W = tf([kW*aW^2], [1 2*aW aW^2]);

Gw = series(G, W);

% --- H-infinity Controller Synthesis ---
nmeas = 1; ncon = 1;
[K, CL, gamma] = hinfsyn(Gw, nmeas, ncon);

disp('H-infinity controller K(s):');
K

% --- Closed-loop Analysis ---
L = series(K, G);
T = feedback(L, 1);
disp('Closed-loop poles:');
disp(pole(T));

figure;
step(T);
title('Closed-loop Step Response');
xlabel('Time');
ylabel('Output');

disp('Done.');
%%
% Sensitivity and complementary sensitivity
L = series(K, G);
S = feedback(1, L);   % Sensitivity
T = feedback(L, 1);   % Complementary sensitivity

figure;
bode(S, T);
legend('Sensitivity S', 'Complementary T');
title('Sensitivity Functions');

% Gain and phase margins
[Gm, Pm, Wcg, Wcp] = margin(L);
disp(['Gain margin: ', num2str(20*log10(Gm)), ' dB']);
disp(['Phase margin: ', num2str(Pm), ' deg']);

%%
%% ================= Controller Validation Section ================= %%
% Assumes:
%   G   - your cleaned plant transfer function (SISO)
%   K   - your H-infinity controller (SISO)

% --- 2. Compute open-loop and closed-loop ---
L = series(K, G);        % Open-loop transfer function
T = feedback(L, 1);      % Closed-loop transfer function (reference to output)
S = feedback(1, L);      % Sensitivity function

% --- 3. Step response of closed-loop ---
figure;
step(T);
grid on;
title('Closed-loop Step Response');
xlabel('Time');
ylabel('Output');

% --- 4. Open-loop Bode plot ---
figure;
bode(L);
grid on;
title('Open-loop Bode Plot');

% --- 5. Sensitivity & Complementary Sensitivity Bode ---
figure;
bode(S, T);
legend('Sensitivity S', 'Complementary Sensitivity T');
grid on;
title('Sensitivity Functions');

% --- 6. Nyquist plot for stability check ---
figure;
nyquist(L);
grid on;
title('Open-loop Nyquist Plot');

% --- 7. Gain & Phase Margins ---
[Gm, Pm, Wcg, Wcp] = margin(L);
fprintf('Gain Margin: %.2f dB at %.4f rad/s\n', 20*log10(Gm), Wcg);
fprintf('Phase Margin: %.2f deg at %.4f rad/s\n', Pm, Wcp);

% --- 8. Frequency response magnitude comparison ---
figure;
freqs = logspace(log10(0.1), log10(max(freq_Hz)), 500);  % Frequency vector
[mag_fit, ~] = bode(G, 2*pi*freqs);                      % Plant magnitude
mag_fit = squeeze(mag_fit);

semilogx(freqs, 20*log10(mag_fit), 'b', 'LineWidth', 1.5); hold on;
[mag_L, ~] = bode(L, 2*pi*freqs);
mag_L = squeeze(mag_L);
semilogx(freqs, 20*log10(mag_L), 'r--', 'LineWidth', 1.5);

xlabel('Frequency [Hz]');
ylabel('Magnitude [dB]');
legend('|G(j\omega)|','|L(j\omega)|');
title('Plant vs Open-loop Magnitude');
grid on;
%%
Kd = c2d(K, 0.005, 'tustin'); % or 'zoh'
Kss = ss(Kd);
A = Kss.A; B = Kss.B; C = Kss.C; D = Kss.D; Ts = Kss.Ts;
save('K_first_try.mat', 'A', 'B', 'C', 'D', 'Ts');

function G_real = force_real_tf(G)
% G_real = force_real_tf(G)
% Removes tiny imaginary parts from numerator/denominator of transfer function G
% Input: G - tf object (SISO or MIMO)
% Output: G_real - tf object with small imaginary parts removed

    tol = 1e-10;  % Tolerance for imaginary parts

    G_real = G;   % Initialize
    for k = 1:length(G.Num)
        % Numerator
        num = G.Num{k};
        num(abs(imag(num)) < tol) = real(num(abs(imag(num)) < tol));

        % Denominator
        den = G.Den{k};
        den(abs(imag(den)) < tol) = real(den(abs(imag(den)) < tol));

        % Update TF
        G_real.Num{k} = num;
        G_real.Den{k} = den;
    end
end