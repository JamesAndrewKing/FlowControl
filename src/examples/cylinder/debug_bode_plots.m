% --- Load frequency response data ---
data_G = load('data_output/nyquist/G_vals.mat');
data_omega = load('data_output/nyquist/omega_range.mat');
omega_data = data_omega.omega_range(:).';
G_vals_data = data_G.G_vals(:).';

% --- Define desired frequency grid ---
data_omega = logspace(-2, 2, 100); % 10^-2 to 10^2 rad/s

% --- Interpolate data to desired omega_range ---
G_vals = interp1(omega_data, G_vals_data, data_omega, 'spline', 'extrap');

% --- Plot Bode plot directly from frequency response data ---
figure;

% Magnitude plot
mag_dB_raw = 20*log10(abs(G_vals));
subplot(2,1,1);
semilogx(data_omega, mag_dB_raw, 'b', 'LineWidth', 1.5);
grid on;
title('Bode Plot: Magnitude');
ylabel('Magnitude (dB)');
xlabel('Frequency (rad/s)');

% Wrapped phase plot with 180 degrees added
phase_deg_wrapped = mod(rad2deg(angle(G_vals)), 360);
subplot(2,1,2);
semilogx(data_omega, phase_deg_wrapped, 'b', 'LineWidth', 1.5);
grid on;
title('Bode Plot: Wrapped Phase');
ylabel('Phase (deg)');
xlabel('Frequency (rad/s)');