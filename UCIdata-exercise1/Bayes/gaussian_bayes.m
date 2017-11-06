clear;
close all;

tic

rng(8); % For reproducibility
load('../datasets.mat');

%Randomize dataset
dataSetSize = size(pimaindiansdiabetes,1);
data_rand = table2array(pimaindiansdiabetes(randperm(dataSetSize),:));
data_rand(:,end) = data_rand(:,end) + 1; %Shift class one value up to help with later calculations

%------Train the gaussian bayes classifier--------------
fprintf('Bayes:\n');

%10fold cross-validate
overallAccuracy = mean(crossval(@bayes_classifier, data_rand));
fprintf('Bayes Overall Accuracy = %4.2f\n',overallAccuracy);

toc