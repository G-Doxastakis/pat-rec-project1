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
    x=((max(class1(:,1))+5-b)/a):0.1:((-b)/a);
    y=a*x+b;
    plot(x,y);
    legend('Letter I','Letter O','Decision limit');
    xlabel('Xaxis variance'); ylabel('Number of Pixels');
    hold off;

end

