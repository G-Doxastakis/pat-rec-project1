clear;
load('../datasets.mat');
dat = datatable2mat(pimaindiansdiabetes); 


X = dat(:,1:8);
Y = dat(:,9);
Mdl = fitcnb(X,Y);

pred = resubPredict(Mdl);
plotconfusion(ind2vec(Y'),ind2vec(pred'));

function dat = datatable2mat(datatable)
    %Converts table with numerical and categorical data to matrix
    c= grp2idx(table2array(datatable(:,end)));
    dat = table2array(datatable(:,1:end-1));
    dat(:,end+1)=c;
end