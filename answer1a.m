clear;
close all;
addpath('./Bayes/');
tic

rng(10); % For reproducibility
load('datasets.mat');

%Randomize dataset
dataSetSize = size(iris,1);
data_rand = iris(randperm(dataSetSize),:); 
data_rand.Class = grp2idx(data_rand.Class);
data_rand = table2array(data_rand);
%------Train the gaussian bayes classifier--------------

%10fold cross-validate
overallAccuracy = mean(crossval(@bayes_classifier, data_rand));
fprintf('Bayes Overall Accuracy = %4.2f\n',overallAccuracy);

%------Train the kNN classifier--------------

%10fold cross-validate
X = data_rand(:,1:end-1);
Y = data_rand(:,end);
Mdl = fitcknn(X,Y);

Mdl.NumNeighbors = 10;
loss = resubLoss(Mdl);
CVMdl = crossval(Mdl,'KFold',10);
kloss = kfoldLoss(CVMdl);

fprintf('kNN Overall Accuracy = %4.2f\n',(1-kloss)*100);

toc