load fisheriris
X = meas;
Y = species;
Mdl = fitcnb(X,Y,'ClassNames',{'setosa','versicolor','virginica'});

isLabels = resubPredict(Mdl);
ConfusionMat = confusionmat(Y,isLabels)