% Clear all variables, close all figures, and clear command window
clc; clear; close all;

% Define input combinations (truth table for 2-input system: A and B)
inputs = [0 0; 0 1; 1 0; 1 1];

% Initialize the output vector to store the results for each input
output = zeros(4, 1);

% Define the weights for the first layer (2 neurons with 2 inputs each)
w1 = [1 -1; -1 1];

% Define the weights for the second layer (1 neuron with 2 inputs)
w2 = [1 1];

% Define the threshold for the first layer
theta1 = 0;

% Define the threshold for the second layer
theta2 = 1;

% Define the activation function as a step function
% The function returns 1 if input >= threshold, otherwise 0
activation = @(x, theta) x >= theta;

% Loop through each input combination
for i = 1:4
    % Compute the output of the first layer neurons
    % Multiply weights (w1) with input vector and apply activation function
    h = activation(w1 * inputs(i, :)', theta1);
    
    % Compute the final output using second layer weights and apply activation
    output(i) = activation(w2 * h, theta2);
end

% Display the truth table with inputs and corresponding output
disp('Input A B | Output');
disp([inputs, output]);