clear;
load('income-usable.mat');
N=8;
for i=1:N
    classes{i}=unique(income(:,i));
    classnum(i)=height(classes{i});
    data(:,i)=grp2idx(table2array(income(:,i)));
end
data = data(1:300,:);

%Create network
% 2 3  5 6 7
% \ /  \ | /
%  1     4  
%   \   /
%     8

dag = zeros(N,N);
dag(2,1)=1; dag(3,1)=1; 
dag(5,4)=1; dag(6,4)=1; dag(7,4)=1;
dag(1,8)=1; dag(4,8)=1;

%Create network
onodes = [2,3,5,6,7];
node_sizes = classnum;
bnet = mk_bnet(dag, node_sizes, 'observed', onodes);
% use random params
for i=1:N
    bnet.CPD{i} = tabular_CPD(bnet, i);
end

%train network
ncases = size(data, 1);
cases = num2cell(data');
engine = jtree_inf_engine(bnet);
bnet = learn_params_em(engine, cases);

%inference
engine = jtree_inf_engine(bnet);
evidence = cell(1,N);
evidence(onodes) = num2cell([1 5 5 2 38]);
[engine, ll] = enter_evidence(engine, evidence);
m = marginal_nodes(engine, 8);