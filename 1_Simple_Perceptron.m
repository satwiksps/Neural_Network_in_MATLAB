clc; clear;
% Define perceptron parameters
numInputs = 10; % Number of inputs
numOutputs = 1; % Number of outputs

% Initialize weights and bias randomly
weights = rand(numInputs, 1);
bias = rand();

% Example input data (10 inputs)
inputData = [0.5, -0.3, 0.8, -0.1, 0.4, -0.5, 0.7, -0.2, 0.6, 0.1];

% Calculate weighted sum
weightedSum = inputData * weights + bias;

% Apply step activation function (thresholding at 0)
if weightedSum > 0
    prediction = 1;
else
    prediction = 0;
end

% Display prediction
fprintf('Prediction: %d\n', prediction);
