clear;
close all;
addpath('./Bayes/');
tic

rng(10); % For reproducibility
load('datasets.mat');

%Randomize dataset
dataSetSize = size(pimaindiansdiabetes,1);
data_rand = table2array(pimaindiansdiabetes(randperm(dataSetSize),:)); 
data_rand(:,end) = data_rand(:,end) +1; %Shift class one value up to help with later calculations

%------Train the gaussian bayes classifier--------------

%10fold cross-validate
bayes_classifier = @(XTRAIN, XTEST)(bayes_classifier(XTRAIN, XTEST, 2));
overallAccuracy = mean(crossval(bayes_classifier, data_rand));
fprintf('Bayes Overall Accuracy = %4.2f\n',overallAccuracy);

%------Train the kNN classifier--------------

%10fold cross-validate
X = data_rand(:,1:end-1);
Y = data_rand(:,end);
Mdl = fitcknn(X,Y);

Mdl.NumNeighbors = 27; % Optimal k value from answer 1a2
loss = resubLoss(Mdl);
CVMdl = crossval(Mdl,'KFold',10);
kloss = kfoldLoss(CVMdl);

fprintf('kNN Overall Accuracy = %4.2f\n',(1-kloss)*100);

toc
