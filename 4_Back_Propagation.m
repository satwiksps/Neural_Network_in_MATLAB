% Clear command window and workspace variables
clc; clear;
% Simple Backpropagation Neural Network in MATLAB with Error Plot for AND
X = [0 0; 0 1; 1 0; 1 1];  % Input patterns (AND problem)
Y = [0; 0; 0; 1];          % Target output for AND

% Parameters
input_neurons = 2; hidden_neurons = 2; output_neurons = 1;
lr = 0.5; epochs = 10;

% Initialize weights and biases
W1 = rand(input_neurons, hidden_neurons) - 0.5;
B1 = rand(1, hidden_neurons) - 0.5;
W2 = rand(hidden_neurons, output_neurons) - 0.5;
B2 = rand(1, output_neurons) - 0.5;

% Activation function
sigmoid = @(x) 1./(1 + exp(-x)); 
dsigmoid = @(x) x .* (1 - x);

% Array to store total error for each epoch
epoch_errors = zeros(epochs, 1);

% Training loop
for epoch = 1:epochs
    total_error = 0;
    for i = 1:size(X, 1)
        % Forward pass
        h_out = sigmoid(X(i, :) * W1 + B1);
        o_out = sigmoid(h_out * W2 + B2);

        % Calculate error and accumulate for plotting
        error = Y(i) - o_out;
        total_error = total_error + sum(error.^2);

        % Backpropagation
        d_o = error .* dsigmoid(o_out);
        d_h = (d_o * W2') .* dsigmoid(h_out);

        % Update weights and biases
        W2 = W2 + lr * h_out' * d_o; B2 = B2 + lr * d_o;
        W1 = W1 + lr * X(i, :)' * d_h; B1 = B1 + lr * d_h;

        % Display result
        fprintf('Epoch %d, Input [%d %d], Target %d, Output %.4f\n', epoch, X(i,1), X(i,2), Y(i), o_out);
    end
    % Store the total error for the current epoch
    epoch_errors(epoch) = total_error;
end

% Plot Error vs Epoch graph
figure;
plot(1:epochs, epoch_errors, '-o');
xlabel('Epoch');
ylabel('Total Error');
title('Error vs Epoch for AND Problem');
grid on;