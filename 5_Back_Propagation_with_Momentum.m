% Clear command window and workspace variables
clc; clear; close all;

% XOR input (binary)
inputs = [0 0; 0 1; 1 0; 1 1];
% XOR targets (binary)
targets = [0; 1; 1; 0];

% Parameters
input_layer_size = 2;  % Two inputs
hidden_layer_size = 2; % Two neurons in hidden layer
output_layer_size = 1; % One output

% Hyperparameters
learning_rate = 0.5;
momentum = 0.9;
epochs = 4000;  % Number of training iterations

% Randomly initialize weights
W1 = rand(input_layer_size, hidden_layer_size) - 0.5;  % Weights from input to hidden layer
W2 = rand(hidden_layer_size, output_layer_size) - 0.5; % Weights from hidden to output layer
bias1 = rand(1, hidden_layer_size) - 0.5;  % Bias for hidden layer
bias2 = rand(1, output_layer_size) - 0.5;  % Bias for output layer

% Initialize momentum terms
V_dW1 = zeros(size(W1));
V_dW2 = zeros(size(W2));
V_db1 = zeros(size(bias1));
V_db2 = zeros(size(bias2));

% Sigmoid activation function and its derivative
sigmoid = @(x) 1 ./ (1 + exp(-x));
sigmoid_deriv = @(x) x .* (1 - x);

% Initialize arrays to store the error, weights, and output for each epoch
errors = zeros(epochs, 1);
W1_values = zeros(epochs, numel(W1));
W2_values = zeros(epochs, numel(W2));
output_evolution = zeros(epochs, size(inputs, 1));

% Training loop
for epoch = 1:epochs
    % Forward propagation
    hidden_input = inputs * W1 + bias1;
    hidden_output = sigmoid(hidden_input);  % Output from hidden layer
    
    final_input = hidden_output * W2 + bias2;
    final_output = sigmoid(final_input);  % Output from output layer

    % Compute the error (output - target)
    error = targets - final_output;
    
    % Store the mean absolute error for this epoch
    errors(epoch) = mean(abs(error));
    % Store the weights and network outputs at each epoch
    W1_values(epoch, :) = W1(:);  % Flatten and store W1
    W2_values(epoch, :) = W2(:);  % Flatten and store W2
    output_evolution(epoch, :) = final_output;  % Store current network output 
    % Backpropagation
    d_final_output = error .* sigmoid_deriv(final_output);  % Delta for output layer
    error_hidden_layer = d_final_output * W2';
    d_hidden_output = error_hidden_layer .* sigmoid_deriv(hidden_output);  % Delta for hidden layer
    % Weight and bias updates with momentum
    dW2 = hidden_output' * d_final_output;
    dW1 = inputs' * d_hidden_output;
    V_dW2 = momentum * V_dW2 + learning_rate * dW2;
    V_dW1 = momentum * V_dW1 + learning_rate * dW1;
    V_db2 = momentum * V_db2 + learning_rate * sum(d_final_output, 1);
    V_db1 = momentum * V_db1 + learning_rate * sum(d_hidden_output, 1);
    W2 = W2 + V_dW2;
    W1 = W1 + V_dW1;
    bias2 = bias2 + V_db2;
    bias1 = bias1 + V_db1;
    % Display error periodically
    if mod(epoch, 1000) == 0
        disp(['Epoch ' num2str(epoch) ' Error: ' num2str(errors(epoch))]);
    end
end

% Plot 1: Error vs. Epoch graph
figure;
plot(1:epochs, errors, 'LineWidth', 1.5);
title('Training Error vs. Epoch');
xlabel('Epoch');
ylabel('Mean Absolute Error');
grid on;

% Plot 2: Weights evolution
figure;
subplot(2, 1, 1);
plot(1:epochs, W1_values, 'LineWidth', 1.5);
title('W1 (Input to Hidden) Weights Evolution');
xlabel('Epoch');
ylabel('W1 Values');
grid on;
subplot(2, 1, 2);
plot(1:epochs, W2_values, 'LineWidth', 1.5);
title('W2 (Hidden to Output) Weights Evolution');
xlabel('Epoch');
ylabel('W2 Values');
grid on;


% Test the final network
final_output = sigmoid(inputs * W1 + bias1) * W2 + bias2;
disp('Final output:');
disp(round(sigmoid(final_output)));  % Rounding the final output to 0 or 1