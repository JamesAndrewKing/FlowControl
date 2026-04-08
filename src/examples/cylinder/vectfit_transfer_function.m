% --- Define omega_range in rad/s ---
omega_range = logspace(-2, 2, 100); % 10^-2 to 10^2 rad/s
s = 1j * omega_range;               % Row vector

% --- Load Data (assume data is already in terms of omega) ---
data_G = load('data_output/nyquist/G_vals.mat');
G_vals = data_G.G_vals(:).';

% --- Initial Poles for VECTFIT ---
n_poles_real = 2;
n_poles_cmplx = 2;
init_poles = -logspace(log10(min(omega_range)), log10(max(omega_range)), n_poles_real);
cplx_freqs = logspace(log10(min(omega_range)), log10(max(omega_range)), n_poles_cmplx);
for k = 1:n_poles_cmplx
    init_poles = [init_poles, -cplx_freqs(k) + 1j*cplx_freqs(k), -cplx_freqs(k) - 1j*cplx_freqs(k)];
end

weight = ones(1, length(s));

% --- VECTFIT Options (unchanged) ---
opts = struct();
opts.relax = 1;
opts.stable = 1;
opts.asymp = 3;
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
residues = SER.C;
d = SER.D;
h = SER.E;

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

% --- Plot Fit Quality (use omega_range) ---
figure;
plot(omega_range, abs(G_vals), 'b', 'DisplayName', 'Original');
hold on;
[mag_fit, ~] = bode(G, omega_range);
plot(omega_range, squeeze(mag_fit), 'r--', 'DisplayName', 'Fitted');
xlabel('\omega [rad/s]');
ylabel('|G(j\omega)|');
legend('Original', 'Fitted');
title('Fit Quality');
grid on;

% --- Weighting Function for Loop Shaping ---
kW = 1.0; aW = 2.0;
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

% Sensitivity and complementary sensitivity
L = series(K, G);
S = feedback(1, L);
T = feedback(L, 1);

figure;
bode(S, T, omega_range);
legend('Sensitivity S', 'Complementary T');
title('Sensitivity Functions');

% Gain and phase margins
[Gm, Pm, Wcg, Wcp] = margin(L);
disp(['Gain margin: ', num2str(20*log10(Gm)), ' dB']);
disp(['Phase margin: ', num2str(Pm), ' deg']);

% --- Controller Validation Section ---
L = series(K, G);
T = feedback(L, 1);
S = feedback(1, L);

figure;
step(T);
grid on;
title('Closed-loop Step Response');
xlabel('Time');
ylabel('Output');

figure;
bode(L, omega_range);
grid on;
title('Open-loop Bode Plot');

figure;
bode(S, T, omega_range);
legend('Sensitivity S', 'Complementary Sensitivity T');
grid on;
title('Sensitivity Functions');

figure;
nyquist(L, omega_range);
grid on;
title('Open-loop Nyquist Plot');

[Gm, Pm, Wcg, Wcp] = margin(L);
fprintf('Gain Margin: %.2f dB at %.4f rad/s\n', 20*log10(Gm), Wcg);
fprintf('Phase Margin: %.2f deg at %.4f rad/s\n', Pm, Wcp);

figure;
freqs = logspace(-2, 2, 500);
[mag_fit, ~] = bode(G, freqs);
mag_fit = squeeze(mag_fit);
semilogx(freqs, 20*log10(mag_fit), 'b', 'LineWidth', 1.5); hold on;
[mag_L, ~] = bode(L, freqs);
mag_L = squeeze(mag_L);
semilogx(freqs, 20*log10(mag_L), 'r--', 'LineWidth', 1.5);
xlabel('\omega [rad/s]');
ylabel('Magnitude [dB]');
legend('|G(j\omega)|','|L(j\omega)|');
title('Plant vs Open-loop Magnitude');
grid on;

Kd = c2d(K, 0.005, 'tustin');
Kss = ss(Kd);
A = Kss.A; B = Kss.B; C = Kss.C; D = Kss.D; Ts = Kss.Ts;
save('K_first_try.mat', 'A', 'B', 'C', 'D', 'Ts');

function G_real = force_real_tf(G)
    tol = 1e-10;
    G_real = G;
    for k = 1:length(G.Num)
        num = G.Num{k};
        num(abs(imag(num)) < tol) = real(num(abs(imag(num)) < tol));
        den = G.Den{k};
        den(abs(imag(den)) < tol) = real(den(abs(imag(den)) < tol));
        G_real.Num{k} = num;
        G_real.Den{k} = den;
    end
end