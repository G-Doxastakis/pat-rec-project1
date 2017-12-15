% Bayes Classifier
% Assumes last column of input TRAIN and TEST matrices is the class
% 
% set covarianceMode to 1 for common diagonal covariance matrix
% set covarianceMode to 2 for common covariance matrix
% set covarianceMode to 3 for a seperate covariance matrix for each class

function testval = bayes_classifier(XTRAIN, XTEST, covarianceMode)
if nargin < 3
    fprintf('No covarianceMode specified : ');
    covarianceMode = 3;
end
fprintf('Fitting Bayes Classifier :\n');

trainDataSize  = size(XTRAIN,1);
testDataSize  = size(XTEST,1);

% Separate all the training samples into separate classes
[C,~,idx] = unique(XTRAIN(:,end));
classes = accumarray(idx,1:trainDataSize,[],@(r){XTRAIN(r,1:end-1)});

% Generate the summary map from all the classes 
classMap = containers.Map(C, classes);

covMap = containers.Map('KeyType','int32','ValueType','any');
meanMap = containers.Map('KeyType','int32','ValueType','any');

% -------------------Calculate covariance and mean-------------------------
switch covarianceMode
    case 1 % Use a common DIAGONAL covariance regardless of class
        fprintf('using COMMON DIAGONAL covariance matrix for all classes :\n');
        
        % First gather all features
        allClassesFeatures = [];
        for i = 1:size(C,1)
            c = C(i);
            allClassesFeatures = [allClassesFeatures; classMap(c)];
        end
        
        % Calculate one covariance matrix regardless of class and multiply
        % it by the diagonal matrix
        covariance = var(allClassesFeatures(:)) * eye (size(allClassesFeatures,2));
        
        for i = 1:size(C,1)
            c = C(i);
            covMap(c) = covariance; % Use the same covariance matrix for all  classes
            meanMap(c) = mean(classMap(c),1);
        end
    case 2 % Use a common covariance matrix regardless of class
        fprintf('using COMMON covariance matrix for all classes :\n');
        
        allClassesFeatures = [];
        for i = 1:size(C,1)
            c = C(i);
            allClassesFeatures = [allClassesFeatures; classMap(c)];
        end
        
        % Calculate one covariance matrix regardless of class
        covariance = cov(allClassesFeatures);
        
        for i = 1:size(C,1)
            c = C(i);
            covMap(c) = covariance;
            meanMap(c) = mean(classMap(c),1);
        end
    case 3 
        % Use seperate covariance matrices for each class
        fprintf('using SEPERATE covariance matrix for each class :\n');

        for i = 1:size(C,1)
            c = C(i);
            if size(classMap(c),1) ~= 0
                covMap(c) = cov(classMap(c)); % Calculate the covariance matrix of THIS class
                meanMap(c) = mean(classMap(c),1);
            end
        end
end

% -------------Find the accuracy of training and testing data--------------

% First get the a priori probabilities for each class
priors = []; % Initialise some space to store the priors
 for i = 1:size(C,1) % Loop  over all classes
     
        c = C(i); % Get the current class
        
        % Get the a priori probability of this class
        priorProb = size(classMap(c),1)/trainDataSize;
        
        % Store it for later use
        priors = cat(2, priors, priorProb); 
 end

% Training ----------------------------------------------------------------

% Initialise some space to store the estimated class for each x
calculatedY = []; 

% Do the estimate! 
for trainIndex=1:trainDataSize % Loop  over all x in the train set
    
    prob = []; % Initialise some space to store the resulting probabilities
    
    % Calculate the a posteriori probabilities for all classes 
    % for current x
    for i = 1:size(C,1) % Loop  over all classes
        
        c = C(i); % Get the current class
        
        % Get the current class' a posteriori probability given current x
        % using mean and covariance for this class.
        % (Note: depending on the covarianceMode chosen, common and  diagonal, 
        % common or seperate covariance matrices might be used. See above.)
        % (Note: no need to normalize by p(x), as its the same everywhere) 
        condProb = mvnpdf(XTRAIN(trainIndex,1:end-1), meanMap(c), covMap(c))*priors(i);
        
        % Put the a posteriori and a priori probabilities of current class
        % for current x side by side for 
        % later comparison
        prob = cat(2, prob, condProb); 
    end
    
    % Get the greatest a posteriori probability (estimate of which class the
    % x falls under)
    [maxval,argmax] = max(prob);
    
    % Store the estimated class for later use
    calculatedY = [calculatedY;argmax];
end

% Now count how many estimates were right by comparing them with the labels
accuratePredictions = (calculatedY == XTRAIN(:,end));
% Get the accuracy percentage
trainingAccuracy = sum(accuratePredictions)*100/size(accuratePredictions,1);
fprintf('Training Accuracy = %4.2f\n',trainingAccuracy);

    
%Testing ------------------------------------------------------------------

% Initialise some space to store the estimated class for each x
calculatedY = [];

% Do the estimate! 
for testIndex=1:testDataSize % Loop  over all x in the train set
    
    prob = []; % Initialise some space to store the resulting probabilities
    
    % Calculate the a posteriori probabilities for all classes 
    % for current x
    for i = 1:size(C,1) % Loop  over all classes
        
        c = C(i); % Get the current class
        
        % Get the current class' a posteriori probability given current x
        % using mean and covariance for this class.
        % (Note: depending on the covarianceMode chosen, common and  diagonal, 
        % common or seperate covariance matrices might be used. See above.)
        % (Note: no need to normalize by p(x), as its the same everywhere) 
        condProb = mvnpdf(XTEST(testIndex,1:end-1), meanMap(c), covMap(c))*priors(i);
        
        % Put the a posteriori and a priori probabilities of current class
        % for current x side by side for 
        % later comparison
        prob = cat(2, prob, condProb);
    end
    
    % Get the greatest a posteriori probability (estimate of which class the
    % x falls under)
    [maxval,argmax] = max(prob);
    
    % Store the estimated class for later use
    calculatedY = [calculatedY;argmax];
end

% Now count how many estimates were right by comparing them with the labels
accuratePredictions = (calculatedY == XTEST(:,end));
% Get the accuracy percentage
testAccuracy = sum(accuratePredictions)*100/size(accuratePredictions,1);
fprintf('Test Accuracy = %4.2f\n\n',testAccuracy);


% return the accuracy percentage to the caller
testval = testAccuracy;