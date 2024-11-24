% Clear command window and workspace variables
clc; clear;

% Define input patterns for the AND gate
inputs = [-1 -1; -1 1; 1 -1; 1 1]; % Each row is an input pattern
% Define target outputs for the AND gate (-1 for false, 1 for true)
targets = [-1; -1; -1; 1];

% Set the learning rate for weight updates
learning_rate = 0.1;
% Set the number of training epochs
epochs = 10;

% Initialize weights randomly for the two inputs
weights = rand(1, 2); 
% Initialize the bias randomly
bias = rand;

% Training loop over epochs
for epoch = 1:epochs
    fprintf('Epoch %d\n', epoch); % Display current epoch
    
    % Loop through each input-target pair
    for i = 1:size(inputs, 1)
        % Compute the weighted sum (input dot product with weights plus bias)
        weighted_sum = weights * inputs(i, :)' + bias;
        
        % Determine the output using the sign activation function
        output = sign(weighted_sum);
        
        % Calculate the error (difference between target and output)
        error = targets(i) - output;
        
        % Update the weights using the Perceptron learning rule
        weights = weights + learning_rate * error * inputs(i, :);
        % Update the bias
        bias = bias + learning_rate * error;
        
        % Display the details of the current training step
        fprintf('Input: [%d %d], Target: %d, Output: %d, Weights: [%f %f], Bias: %f\n', ...
            inputs(i, 1), inputs(i, 2), targets(i), output, weights, bias);
    end
end

% Display the final weights and bias after training
disp('Final weights:');
disp(weights);
disp('Final bias:');
disp(bias);

