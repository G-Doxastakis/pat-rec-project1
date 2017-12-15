function [ y ] = testPerceptron( x , W )
    x=[x ones(size(x,1),1)];
    y=sign(x*W);
end

