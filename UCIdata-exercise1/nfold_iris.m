clear;
iris = datatable2mat(dataset2table(dataset('File', 'iris.data',...
    'Delimiter', ',', 'ReadVarNames', false)));
folds = 10;
iris = nFoldDataset(iris,folds);
for K=1:9
    for f=1:folds
         err(f)=knnclassifier(iris(f).train,iris(f).val, K);
    end
    sub = subplot(3,3, K);
    plot(err); 
    title(['Plot for K = ', num2str(K)])
end

function dat = datatable2mat(datatable)
    %Converts table with numerical and categorical data to matrix
    c= grp2idx(table2array(datatable(:,end)));
    dat = table2array(datatable(:,1:end-1));
    dat(:,end+1)=c;
end