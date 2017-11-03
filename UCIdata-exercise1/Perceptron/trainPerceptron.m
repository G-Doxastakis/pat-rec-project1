function [ W,epochs] = trainPerceptron( x, y, rate )
    x=[x ones(size(x,1),1)];
    W=rand(1,size(x,2))';
    yapr=zeros(size(x,1),1);
    epochs=0;
    while(~isequal(y,yapr))
        epochs=epochs+1;
        for i=1:size(x,1)
            yapr(i)=sign(x(i,:)*W);
            W=W+(rate*(y(i)-yapr(i)))*(x(i,:)');
        end
    end
end

