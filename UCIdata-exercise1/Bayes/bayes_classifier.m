%Bayes classifier
%Assumes last column of input TRAIN and TEST matrices is the class
%set covarianceMode to 1 for common diagonal covariance matrix
%set covarianceMode to 2 for common covariance matrix
%set covarianceMode to 3 for a seperate covariance matrix for each class

function testval = bayes_classifier(XTRAIN, XTEST, covarianceMode)
if nargin < 3
    fprintf('No covarianceMode specified : ');
    covarianceMode = 3;
end
fprintf('Fitting Bayes Classifier :\n');

trainDataSize  = size(XTRAIN,1);
testDataSize  = size(XTEST,1);

%Separate all the training samples into separate classes
[C,~,idx] = unique(XTRAIN(:,end));
classes = accumarray(idx,1:trainDataSize,[],@(r){XTRAIN(r,1:end-1)});

%Generate the summary map from all the classes 
classMap = containers.Map(C, classes);

covMap = containers.Map('KeyType','int32','ValueType','any');
meanMap = containers.Map('KeyType','int32','ValueType','any');

switch covarianceMode
    case 1
        %Calculate the common diagonal covariance and mean of all the features for all the
        %classes
        fprintf('using COMMON DIAGONAL covariance matrix for all classes :\n');
        
        allClassesFeatures = [];
        for i = 1:size(C,1)
            c = C(i);
            allClassesFeatures = [allClassesFeatures; classMap(c)];
        end
        
        for i = 1:size(C,1)
            c = C(i);
            covMap(c) = cov(allClassesFeatures(:)) * eye (size(allClassesFeatures,2));
            meanMap(c) = mean(allClassesFeatures,1);
        end
    case 2
        %Calculate the common covariance and mean of all the features for all the
        %classes
        fprintf('using COMMON covariance matrix for all classes :\n');
        
        allClassesFeatures = [];
        for i = 1:size(C,1)
            c = C(i);
            allClassesFeatures = [allClassesFeatures; classMap(c)];
        end
        for i = 1:size(C,1)
            c = C(i);
            covMap(c) = cov(allClassesFeatures);
            meanMap(c) = mean(allClassesFeatures,1);
        end
    case 3 
        %Calculate the covariance and mean of all the features for all the
        %classes seperately
        fprintf('using SEPERATE covariance matrix for each class :\n');

        for i = 1:size(C,1)
            c = C(i);
            if size(classMap(c),1) ~= 0
                covMap(c) = cov(classMap(c));
                meanMap(c) = mean(classMap(c),1);
            end
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
            condProb = mvnpdf(XTRAIN(trainIndex,1:end-1), meanMap(c), covMap(c));
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
fprintf('Training Accuracy = %4.2f\n',trainingAccuracy);
    
%Testing
calculatedY = [];
for testIndex=1:testDataSize
    prob = [];
    for i = 1:size(C,1)
        c = C(i);
        if size(classMap(c),1) ~= 0 
            condProb = mvnpdf(XTEST(testIndex,1:end-1), meanMap(c), covMap(c));
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
fprintf('Test Accuracy = %4.2f\n\n',testAccuracy);
testval = testAccuracy;