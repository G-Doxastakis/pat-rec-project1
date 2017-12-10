clear;
close all;

tic

rng(8); % For reproducibility
load('../datasets.mat');

%Randomize dataset
dataSetSize = size(pimaindiansdiabetes,1);
data_rand = table2array(pimaindiansdiabetes(randperm(dataSetSize),:));
data_rand(:,end) = data_rand(:,end) + 1; %Shift class one value up to help with later calculations


%set mode to 1 for common diagonal covariance matrix
%set mode to 2 for common covariance matrix
%set mode to 3 for a seperate covariance matrix for each class
mode = 3;
classifier = @(XTRAIN, XTEST)(bayes_classifier(XTRAIN, XTEST, mode));
%10fold cross-validate
overallAccuracy = mean(crossval(classifier, data_rand));
fprintf('Bayes Overall Accuracy = %4.2f\n',overallAccuracy);

toc