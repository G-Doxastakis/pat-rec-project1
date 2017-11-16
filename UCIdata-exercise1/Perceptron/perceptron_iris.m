clear;
close all;
load('../datasets.mat');
classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];
iris = datatable2mat(iris);
class1=iris(iris(:,5) == 1, 3:4);
class2=iris(iris(:,5) == 2, 3:4);
class3=iris(iris(:,5) == 3, 3:4);

subplot(131);
InputsTrain = [class1;class2];
OutputTrain = [ones(50,1);-ones(50,1)];
r = randperm(100)';
[W1,ep1] = trainPerceptron(InputsTrain(r,:),OutputTrain(r,:),0.1);
perceptronVisualization(InputsTrain,OutputTrain,W1);
legend(classes(1),classes(2),'Decision limit');
xlabel('petal length (cm)'); ylabel('petal width (cm)');
xlim([0,3]);ylim([0,8]);

subplot(132);
InputsTrain = [class3;class1];
OutputTrain = [ones(50,1);-ones(50,1)];
r = randperm(100)';
[W2,ep2] = trainPerceptron(InputsTrain(r,:),OutputTrain(r,:),0.1);
perceptronVisualization(InputsTrain,OutputTrain,W2);
legend(classes(3),classes(1),'Decision limit');
xlabel('petal length (cm)'); ylabel('petal width (cm)');
xlim([0,3]);ylim([0,8]);

subplot(133);
InputsTrain = [class2;class3];
OutputTrain = [ones(50,1);-ones(50,1)];
r = randperm(100)';
[W3,ep3] = trainPerceptron(InputsTrain(r,:),OutputTrain(r,:),0.1);
perceptronVisualization(InputsTrain,OutputTrain,W3);
legend(classes(2),classes(3),'Decision limit');
xlabel('petal length (cm)'); ylabel('petal width (cm)');
xlim([0,3]);ylim([0,8]);

function dat = datatable2mat(datatable)
    %Converts table with numerical and categorical data to matrix
    c= grp2idx(table2array(datatable(:,end)));
    dat = table2array(datatable(:,1:end-1));
    dat(:,end+1)=c;
end