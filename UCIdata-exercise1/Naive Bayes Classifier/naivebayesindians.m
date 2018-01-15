clear;
rng(10);
load('../datasets.mat');
dat = datatable2mat(pimaindiansdiabetes); 
folds = nFoldDataset(array2table(dat),10);

for i=1:10
    dat = table2array(folds(i).train);
    class1 = dat(dat(:,9) == 1, 1:8);
    mu1 = mean(class1);
    sig1 = var(class1).*eye(8);
    class2 = dat(dat(:,9) == 2, 1:8);
    mu2 = mean(class2);
    sig2 = var(class2).*eye(8);
    
    dat = table2array(folds(i).val);
    p=[mvnpdf(dat(:,1:8),mu1,sig1) mvnpdf(dat(:,1:8),mu2,sig2)];
    [x,pred] = max(p,[],2);
    acc(i) =1-(sum(abs(pred-dat(:,9)))/length(pred));   
end
disp(['Accuracy: ' num2str(mean(acc))]);
function dat = datatable2mat(datatable)
    %Converts table with numerical and categorical data to matrix
    c= grp2idx(table2array(datatable(:,end)));
    dat = table2array(datatable(:,1:end-1));
    dat(:,end+1)=c;
end