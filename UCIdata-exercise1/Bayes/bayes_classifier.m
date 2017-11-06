%Bayes classifier
%Assumes last column of input TRAIN and TEST matrices is the class

function testval = bayes_classifier(XTRAIN, XTEST)
%fprintf('Fitting Bayes Classifier :\n');

trainDataSize  = size(XTRAIN,1);
testDataSize  = size(XTEST,1);

%Separate all the training samples into separate classes
[C,~,idx] = unique(XTRAIN(:,end));
classes = accumarray(idx,1:trainDataSize,[],@(r){XTRAIN(r,1:end-1)});

%Generate the summary map from all the classes 
classMap = containers.Map(C, classes);

%Calculate the std deviation and mean of all the features for all the
%classes
stdMap = containers.Map('KeyType','int32','ValueType','any');
meanMap = containers.Map('KeyType','int32','ValueType','any');

for i = 1:size(C,1)
    c = C(i);
    if size(classMap(c),1) ~= 0
        stdMap(c) = std(classMap(c),1,1);
        meanMap(c) = mean(classMap(c),1);
    end
end

%-------------Find the accuracy of training and testing data------

%Training
calculatedY = [];
for trainIndex=1:trainDataSize
    prob = [];
    for i = 1:size(C,1)
        c = C(i);
        if size(classMap(c),1) ~= 0 
            condProb = normpdf(XTRAIN(trainIndex,1:end-1), meanMap(c),stdMap(c));
            condProb = prod(condProb);
            priorProb = size(classMap(c),1)/trainDataSize;
            prob = cat(2,prob,condProb*priorProb);
        else
            prob = cat(2,prob,0);
        end
    end
    [maxval,argmax] = max(prob);
    calculatedY = [calculatedY;argmax];
end

accuratePredictions = (calculatedY == XTRAIN(:,end));
trainingAccuracy = sum(accuratePredictions)*100/size(accuratePredictions,1);
%fprintf('Training Accuracy = %4.2f\n',trainingAccuracy);
    
%Testing
calculatedY = [];
for testIndex=1:testDataSize
    prob = [];
    for i = 1:size(C,1)
        c = C(i);
        if size(classMap(c),1) ~= 0 
            condProb = normpdf(XTEST(testIndex,1:end-1), meanMap(c),stdMap(c));
            condProb = prod(condProb);
            priorProb = size(classMap(c),1)/trainDataSize;
            prob = cat(2,prob,condProb*priorProb);
        else
            prob = cat(2,prob,0);
        end
    end
    [maxval,argmax] = max(prob);
    calculatedY = [calculatedY;argmax];
end

accuratePredictions = (calculatedY == XTEST(:,end));
testAccuracy = sum(accuratePredictions)*100/size(accuratePredictions,1);
%fprintf('Test Accuracy = %4.2f\n\n',testAccuracy);
testval = testAccuracy;