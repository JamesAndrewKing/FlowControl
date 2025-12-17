function S = load_sparse_from_python(matfilename, varprefix)
% LOAD_SPARSE_FROM_PYTHON Load a sparse matrix saved in CSR format from Python
%
% Inputs:
%   matfilename - string, path to the .mat file (e.g., 'E_sparse.mat')
%   varprefix   - string, prefix used for variable names in the mat file (e.g., 'E')
%
% Output:
%   S - MATLAB sparse matrix reconstructed

    % Load variables
    data = load(matfilename, [varprefix '_data']);
    indices = load(matfilename, [varprefix '_indices']);
    indptr = load(matfilename, [varprefix '_indptr']);
    shape = load(matfilename, [varprefix '_shape']);
    
    data = data.([varprefix '_data']);
    indices = indices.([varprefix '_indices']);
    indptr = indptr.([varprefix '_indptr']);
    shape = shape.([varprefix '_shape']);
    
    % Convert all to double (MATLAB indexing, logical, and sparse require doubles)
    data = double(data);
    indices = double(indices);
    indptr = double(indptr);
    shape = double(shape);
    
    % Number of rows and cols
    nrows = shape(0+1);  % Python is 0-based, MATLAB 1-based indexing, just regular indexing here
    ncols = shape(1+1);
    
    % Reconstruct row indices from indptr (CSC format) to row and col indices:
    % Note: Python CSR = Compressed Sparse Row format
    %   indptr: length = nrows+1, indices: column indices for nonzeros, data = values
    % MATLAB sparse requires (row, col, value)
    % So:
    %   For each row r:
    %       col = indices[indptr[r]:indptr[r+1]-1]
    %       val = data[indptr[r]:indptr[r+1]-1]
    %       row = r for these entries
    
    % Build vectors for rows, columns, and values
    rows = zeros(length(data), 1);
    cols = zeros(length(data), 1);
    
    % Populate rows and cols
    idx = 1;
    for r = 1:nrows
        start_ind = indptr(r) + 1;  % +1 for MATLAB indexing
        end_ind = indptr(r+1);
        count = end_ind - start_ind + 1;
        if count > 0
            rows(idx:idx+count-1) = r;
            cols(idx:idx+count-1) = indices(start_ind:end_ind) + 1;  % +1 for MATLAB indexing
            idx = idx + count;
        end
    end
    
    % Construct sparse matrix
    S = sparse(rows, cols, data, nrows, ncols);
end


A = load_sparse_from_python('data_output/operators/A_sparse.mat', 'A');
E = load_sparse_from_python('data_output/operators/E_sparse.mat', 'E');
Re = str2double(fileread("data_output/current_Re.txt"));
%%
neig = 10;          % number of eigenvalues per shift
opts.tol = 1e-8;
opts.maxit = 1000;

% Frequencies to scan along imaginary axis
% omega_list = 1i * linspace(0, 5, 10);
omega_list1 = -0.1 +1i * linspace(0, 5, 5);
omega_list2 = 0.1 + 1i * linspace(0, 5, 5);
omega_list = [omega_list1, omega_list2];

% all_lambda = [];      % store all eigenvalues found
% all_vecs = {};        % store eigenvectors in cell array, aligned with all_lambda entries
% 
% for omega = omega_list
%     target = 1i * omega;  % shift along imaginary axis
%     try
%         [vecs, vals] = eigs(A, E, neig, target, opts);
%         lambda = diag(vals);
%         all_lambda = [all_lambda; lambda];  %#ok<AGROW>
%         % Store eigenvectors in columns, cell per batch
%         all_vecs{end+1} = vecs; 
%     catch ME
%         warning('eigs failed for omega = %f + %fi\n%s', real(target), imag(target), ME.message);
%     end
% end
% 
% % Find eigenvalue with largest real part
% [~, idx_max] = max(real(all_lambda));
% 
% % To find which batch and column this eigenvalue corresponds to:
% count = 0;
% found_vec = [];
% for batch_idx = 1:length(all_vecs)
%     batch_size = size(all_vecs{batch_idx}, 2);
%     if idx_max <= count + batch_size
%         col_idx = idx_max - count;
%         found_vec = all_vecs{batch_idx}(:, col_idx);
%         break;
%     else
%         count = count + batch_size;
%     end
% end

% % Now found_vec is the eigenvector corresponding to the eigenvalue with the largest real part
% disp(['Largest real part eigenvalue: ', num2str(all_lambda(idx_max))]);
% save('slow_eigenvector.mat', 'found_vec');
% --- Collect all right and left eigenpairs first ---
% filepath: your_script.m
right_eigs = []; % struct array: .lambda, .vec
left_eigs  = []; % struct array: .lambda, .vec

% --- Step 1: Compute right eigenpairs for each omega in omega_list ---
for omega = omega_list
    target = omega;
    try
        [vecs, vals] = eigs(A, E, neig, target, opts);
        lambda = diag(vals);
        for j = 1:length(lambda)
            right_eigs(end+1).lambda = lambda(j);
            right_eigs(end).vec = vecs(:, j);
        end
    catch ME
        warning('Right eigs failed for omega = %f + %fi\n%s', real(target), imag(target), ME.message);
    end
end

% --- Step 2: For each right eigenvalue, compute the corresponding left eigenvector ---
for i = 1:length(right_eigs)
    target = conj(right_eigs(i).lambda); % Use conjugate as target for left eigenproblem
    try
        [lvecs, lvals] = eigs(A', E', 1, target, opts); % Only need 1 eigenpair per target
        left_eigs(i).lambda = diag(lvals);
        left_eigs(i).vec = lvecs;
    catch ME
        warning('Left eigs failed for target = %f + %fi\n%s', real(target), imag(target), ME.message);
        left_eigs(i).lambda = NaN;
        left_eigs(i).vec = NaN;
    end
end

%%

% --- Now match right and left eigenpairs directly by construction ---
eig_data = [];
tol = 1e-4;  % Tolerance for duplicate removal
match_tol = 1e-4; % Tolerance for matching left/right eigenvalues

used_left = false(1, length(left_eigs));
for i = 1:length(right_eigs)
    % Find left eigenvalue closest to conj(right eigenvalue)
    diffs = abs([left_eigs.lambda] - conj(right_eigs(i).lambda));
    diffs(used_left) = Inf; % Don't match already used left eigenvectors
    [min_dist, idx_left] = min(diffs);
    if min_dist < match_tol && ~isnan(left_eigs(idx_left).lambda)
        eig_data(end+1).lambda = right_eigs(i).lambda;
        eig_data(end).vec = right_eigs(i).vec;
        eig_data(end).lvec = left_eigs(idx_left).vec;
        used_left(idx_left) = true;
    end
end

% Remove duplicates by eigenvalue (within tol)
% unique_idx = true(1, length(eig_data));
% for i = 1:length(eig_data)
%     for j = i+1:length(eig_data)
%         if abs(eig_data(i).lambda - eig_data(j).lambda) < tol
%             unique_idx(j) = false;
%         end
%     end
% end
% eig_data = eig_data(unique_idx);
% Remove duplicates by eigenvalue (within tol)
all_lambda = [eig_data.lambda];
[~, unique_idx] = unique(round(all_lambda/tol)*tol, 'stable');
eig_data = eig_data(unique_idx);

% Sort eig_data by real part of eigenvalues (descending)
[~, sort_idx] = sort(real([eig_data.lambda]), 'descend');
eig_data = eig_data(sort_idx);

% Normalize eigenvectors and biorthogonalize
for k = 1:length(eig_data)
    norm_factor_right = sqrt(eig_data(k).vec' * E * eig_data(k).vec);
    eig_data(k).vec = eig_data(k).vec / norm_factor_right;
    norm_factor_left = eig_data(k).lvec' * E * eig_data(k).vec;
    eig_data(k).lvec = eig_data(k).lvec / conj(norm_factor_left);
end

% --- Now all outputs are in sorted order ---
% Display the top eigenvalue
top_eig = eig_data(1);
disp(['Largest real part eigenvalue: ', num2str(top_eig.lambda)]);

% Save corresponding eigenvector
found_vec = top_eig.vec;
found_lvec = top_eig.lvec;
%%
% Check right eigenvector residual: A*phi - lambda*E*phi
res_right = norm(A * found_vec - top_eig.lambda * (E * found_vec))/norm(found_lvec);
disp(['Right eigenvector relative residual norm: ', num2str(res_right)]);

% Check left eigenvector residual: A'*psi - conj(lambda)*E'*psi
res_left = norm(found_lvec' * A - top_eig.lambda * (found_lvec' * E))/norm(found_lvec);
disp(['Left eigenvector relative residual norm: ', num2str(res_left)]);

% Check biorthogonality: psi^H * E * phi
biorth = found_lvec' * E * found_vec;
disp(['Biorthogonality (psi^H E phi): ', num2str(biorth)]);

% Save all sorted outputs
save('data_output/slow_eigenvector.mat', 'found_vec');
save('data_output/slow_lefteigenvector.mat', 'found_lvec');
save('data_output/eig_data.mat', 'eig_data');

%%
all_lambda = [eig_data.lambda];
figure('Position', [100, 100, 800, 400]);  % wider figure: [left bottom width height]
plot(real(all_lambda), imag(all_lambda), 'bo', 'MarkerSize', 8, 'LineWidth', 1.5);
grid on;
xlabel('Real Part');
ylabel('Imaginary Part');
title('Eigenvalues near imaginary axis');

hold on;
% xlim([-0.7 0.1]);  % Set x-axis limits from -1 to 0.1
xlim([-0.1 0.2]);
% xlim([-2.0 0.05]);
ylim([0 5]);

plot(xlim, [0 0], 'k--'); % horizontal zero line
plot([0 0], ylim, 'k--'); % vertical zero line

% Highlight the eigenvalue with largest real part
plot(real(all_lambda(1)), imag(all_lambda(1)), 'ro', 'MarkerSize', 12, 'LineWidth', 2);

hold off;

%%
all_lambda = [eig_data.lambda];
figure('Position', [100, 100, 800, 400]);  % wider figure: [left bottom width height]
plot(real(all_lambda), imag(all_lambda), 'bo', 'MarkerSize', 8, 'LineWidth', 1.5);
grid on;
xlabel('Real Part');
ylabel('Imaginary Part');
title('Eigenvalues near imaginary axis');

hold on;
% xlim([-0.7 0.1]);  % Set x-axis limits from -1 to 0.1
xlim([-0.1 0.05]);
% xlim([-2.0 0.05]);
ylim([0 5]);

plot(xlim, [0 0], 'k--'); % horizontal zero line
plot([0 0], ylim, 'k--'); % vertical zero line

% Highlight the eigenvalue with largest real part
plot(real(all_lambda(1)), imag(all_lambda(1)), 'ro', 'MarkerSize', 12, 'LineWidth', 2);

hold off;
