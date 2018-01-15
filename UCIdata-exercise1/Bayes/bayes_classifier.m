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

% Separate all the training samples into separate classes
[C,~,idx] = unique(XTRAIN(:,end));
classes = accumarray(idx,1:trainDataSize,[],@(r){XTRAIN(r,1:end-1)});

covMap = containers.Map('KeyType','int32','ValueType','any');
meanMap = containers.Map('KeyType','int32','ValueType','any');

X = XTRAIN(:, 1:end-1);
% -------------------Calculate covariance and mean-------------------------
switch covarianceMode
    case 1 % Use a common DIAGONAL covariance regardless of class
        fprintf('using COMMON DIAGONAL covariance matrix for all classes :\n');

        % Calculate one covariance matrix regardless of class and multiply
        % it by the diagonal matrix
        variance = var(X(:)) * eye (size(X,2));
        for i = 1:size(C,1)
            c = C(i);
            covMap(c) = variance; % Use the same covariance matrix for all  classes
            meanMap(c) = mean(X((XTRAIN(:, end) == c), :),1);
        end
    case 2 % Use a common covariance matrix regardless of class
        fprintf('using COMMON covariance matrix for all classes :\n');
              
        % Calculate one covariance matrix regardless of class
        covariance = cov(X)+0.0001 *eye(size(X,2));
        for i = 1:size(C,1)
            c = C(i);
            covMap(c) = covariance;
            meanMap(c) = mean(X((XTRAIN(:, end) == c), :),1);
        end
    case 3 
        % Use seperate covariance matrices for each class
        fprintf('using SEPERATE covariance matrix for each class :\n');

        for i = 1:size(C,1)
            c = C(i);
            sigma = cov(X((XTRAIN(:, end) == c), :))+0.0001 *eye(size(X((XTRAIN(:, end) == c), :),2));
            covMap(c) = sigma; % Calculate the covariance matrix of THIS class
            meanMap(c) = mean(X((XTRAIN(:, end) == c), :),1);
        end
end

% -------------Find the accuracy of training and testing data--------------

% First get the a priori probabilities for each class
priors = []; % Initialise some space to store the priors
 for i = 1:size(C,1) % Loop  over all classes
     
        c = C(i); % Get the current class
        
        % Get the a priori probability of this class
        priorProb = size(X((XTRAIN(:, end) == c), :),1)/trainDataSize;
        
        % Store it for later use
        priors = cat(2, priors, priorProb); 
 end

% Training ----------------------------------------------------------------

% Initialise some space to store the posterior probabilities for each x for
% each class
posteriors = zeros(size(X,1), size(C,1)); 

for i = 1:size(C,1) % Loop  over all classes
    
    % Get the current class' a posteriori probability for each x
    % using mean and covariance for this class.
    % (Note: depending on the covarianceMode chosen, common and  diagonal, 
    % common or seperate covariance matrices might be used. See above.)
    % (Note: no need to normalize by p(x), as its the same everywhere) 
    posteriors(:,i) = mvnpdf(X, meanMap(C(i)), covMap(C(i)))*priors(i);
end

% Get the greatest a posteriori probabilities (estimate of which class each
% x falls under)
[~, greatestPosteriors] = max(posteriors,[],2);

% Now count how many estimates were right by comparing them with the labels
accuratePredictions = (greatestPosteriors == XTRAIN(:,end));
% Get the accuracy percentage
trainingAccuracy = sum(accuratePredictions)*100/size(accuratePredictions,1);
fprintf('Training Accuracy = %4.2f\n',trainingAccuracy);
    
%Testing ------------------------------------------------------------------

% Initialise some space to store the posterior probabilities for each x for
% each class
posteriors = zeros(size(XTEST,1), size(C,1)); 
X = XTEST(:,1:end-1);

for i = 1:size(C,1) % Loop  over all classes
    
    % Get the current class' a posteriori probability for each x
    % using mean and covariance for this class.
    % (Note: depending on the covarianceMode chosen, common and  diagonal, 
    % common or seperate covariance matrices might be used. See above.)
    % (Note: no need to normalize by p(x), as its the same everywhere) 
    posteriors(:,i) = mvnpdf(X, meanMap(C(i)), covMap(C(i)))*priors(i);
end

% Get the greatest a posteriori probabilities (estimate of which class each
% x falls under)
[~, greatestPosteriors] = max(posteriors,[],2);

% Now count how many estimates were right by comparing them with the labels
accuratePredictions = (greatestPosteriors == XTEST(:,end));
% Get the accuracy percentage
testAccuracy = sum(accuratePredictions)*100/size(accuratePredictions,1);
fprintf('Test Accuracy = %4.2f\n\n',testAccuracy);


% return the accuracy percentage to the caller
testval = testAccuracy;