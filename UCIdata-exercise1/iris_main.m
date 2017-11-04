clear;
close all;

upToK = 5; % up to how many neighbors?
upToFold = 10; % up to how many folds?


tic 

filename = 'iris.data';
DATA = dataset('File', filename, 'Delimiter', ',', 'ReadVarNames', false);

%Assign a unique number classes
u = unique(DATA(:, 5));
u = dataset2cell(u);
u = u(2 : end, :);
p = [1 : size(u, 1)];
C = containers.Map(u, p);
t = DATA(:, 5);
t = dataset2cell(t);
t = t(2 : end);
DATA.Var5 = C.values(t);
z = DATA(:, 5);
z = dataset2cell(z);
z = z(2 : end);
z = cell2mat(z);
z = mat2dataset(z);
DATA(:, 5) = z;
DATA = double(DATA);

%Generate random permutation for division into test and train data
nrows = size(DATA, 1);
randrows = randperm(nrows);

%meanaccuracy contains the accuracy data
%rows in meanaccuracy denote fold
%and column denote the corresponding value of K
meanaccuracy = zeros(upToFold, upToK);
X = DATA;
k = 2:upToFold+1;
e = zeros(upToFold, 1);
for K = 1 : upToK
    for fold = 2 : upToFold+1
        for chunk = 1 : fold
            chunksize = floor(nrows/fold);
            x = (chunk - 1) * chunksize + 1;
            y = chunk * chunksize;
            testdata = X(randrows(x:y), :);
            if chunk == 1
                traindata = X(randrows(y + 1:end), :);
            elseif chunk == fold
                traindata = X(randrows(1 : x-1), :);
            else
                traindata = X(randrows(1, x-1:y+1, end), :);
            end
            currentacc = knnclassifier(traindata, testdata, K);
            s(chunk) = currentacc;
        end
        meanaccuracy(fold - 1, K) = mean(s);
        out(fold - 1) = mean(s);
        e(fold - 1) = std(s);      
    end
    sub = subplot(3,3, K);
    errorbar(k, out, e); 
    
    title(['Plot for K = ', num2str(K)])
end

toc
