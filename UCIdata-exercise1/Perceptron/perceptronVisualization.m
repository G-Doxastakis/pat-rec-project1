function [ output_args ] = perceptronVisualization( Inputs,Output,net )
%PERCEPTRONVISUALIZATION Summary of this function goes here
%   Detailed explanation goes here
    dat=[Inputs Output];
    class1=dat(dat(:,3) == 1, :);
    class2=dat(dat(:,3) == -1, :);
    scatter(class1(:,2),class1(:,1),'*r');
    hold on;
    scatter(class2(:,2),class2(:,1),'ob');
    a=(net(2)/(-net(1)));
    b=(net(3)/(-net(1)));
    refline([a,b])
    %legend('Class1','Class2','Decision limit');
    %xlabel('Feature 1'); ylabel('Feature 2');
    hold off;

end

