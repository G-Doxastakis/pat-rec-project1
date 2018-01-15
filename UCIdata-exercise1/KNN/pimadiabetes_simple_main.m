clear;
close all;

upToK = 30; % up to how many neighbors?

tic 

load('datasets.mat');

X = pimaindiansdiabetes(:,1:end-1);
Y = table2array(pimaindiansdiabetes(:,end));
rng(10); % For reproducibility
Mdl = fitcknn(X,Y);

g = zeros(upToK,2);
for K = 1:upToK
    Mdl.NumNeighbors = K;
    loss = resubLoss(Mdl);
    CVMdl = crossval(Mdl,'KFold',10);
    kloss = kfoldLoss(CVMdl);
    g(K,1) = K;
    g(K,2) = kloss*100;
end

plot(g(:,1), g(:,2));
title('Σύγκριση σφάλματος ταξινόμησης ινδιάνων Pima με διαφορετικές τιμές Κ');
xlabel('Επιλεγμένος αριθμός κοντινότερων γειτόνων: ');
ylabel('Σφάλμα %:');

[minError, ind] = min(g(:,2));
bestK = g(ind,1);
txt2 = strcat(strcat('Βέλτιστος αριθμός K = ',num2str(bestK)),' (σφάλμα: ',num2str(minError), ') \rightarrow');
text(bestK,minError,txt2,'Color','red','HorizontalAlignment','right');

toc
