clear;
close all;

upToK = 30; % up to how many neighbors?

tic 

load('datasets.mat');

X = iris(:,1:end-1);
Y = cellstr(table2array(iris(:,end)));
rng(10); % For reproducibility
Mdl = fitcknn(X,Y);

correlation = zeros(upToK,2);
for K = 1:upToK
    Mdl.NumNeighbors = K;
    loss = resubLoss(Mdl);
    CVMdl = crossval(Mdl,'KFold',10);
    kloss = kfoldLoss(CVMdl);
    correlation(K,1) = K;
    correlation(K,2) = kloss*100;
end
plot(correlation(:,1), correlation(:,2));
title('�������� ��������� ����������� ����� Iris �� ������������ ����� �');
xlabel('����������� ������� ������������ ��������: ');
ylabel('������ %:');

[minError, ind] = min(correlation(:,2));
bestK = correlation(ind,1);
txt2 = strcat(strcat('\leftarrow ', '��������� ������� K = ',num2str(bestK)),' (������: ',num2str(minError), ')');
text(bestK,minError,txt2,'Color','red','HorizontalAlignment','left');

toc
