clear;
close all;

tic

rng(8); % For reproducibility
load('../datasets.mat');

% Randomize dataset
dataSetSize = size(pimaindiansdiabetes,1);
data_rand = table2array(pimaindiansdiabetes(randperm(dataSetSize),:));
data_rand(:,end) = data_rand(:,end) + 1; %Shift class one value up to help with later calculations

accuracies = [];

% Calculate accuracy rates for all different covariance modes using 
% 10fold cross-validation method
for i = 1:3 % Loop over all covariance modes
    
    % mode 1 for common diagonal covariance matrix
    % mode 2 for common covariance matrix
    % mode 3 for a seperate covariance matrix for each class
    covarianceMode = i;
    
    % Wrap the classifier function inside another so we can pass the mode as a
    % parameter and keep matlab happy.
    % That is because matlab's crossval function takes as first
    % parameter a function who's signature is as follows:
    % @(XTRAIN, XTEST)(success_rate)
    % The first being the train data set and the seconf the test data set.
    % Our bayes_classifier accepts a third parameter as well.
    classifier = @(XTRAIN, XTEST)(bayes_classifier(XTRAIN, XTEST, covarianceMode));

    % 10fold cross-validate
    overallAccuracy = mean(crossval(classifier, data_rand));
    fprintf('Bayes Overall Accuracy = %4.2f\n',overallAccuracy);
    accuracies = [accuracies; overallAccuracy]; 
end
figure;
x = categorical({'common diagonal covariance matrix','common covariance matrix','seperate covariance matrix for each class'});
bar(x, accuracies);
title('Comparison of classifier accuracy with different covariance modes');
xlabel('Covariance Mode:'); % x-axis label
ylabel('Accuracy %:'); % y-axis label
for i=1:3
    text(x(i),accuracies(i),num2str(accuracies(i),'%0.2f'),...
               'HorizontalAlignment','center',...
               'VerticalAlignment','bottom')
end
toc