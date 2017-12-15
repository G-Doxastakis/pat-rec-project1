function [ dataset ] = nFoldDataset( database, folds )
%NFOLDDATASET Summary of this function goes here
%   Detailed explanation goes here
    [m,n] = size(database);
    database(randperm(m),:)=database;
    seg=uint32(floor(m/folds));
    for i=1:folds
        validx=((folds-i)*seg)+1:(folds-i+1)*seg;
        dataset(i).train=database;
        dataset(i).train(validx,:)=[];
        dataset(i).val=database(validx,:);
    end
end

