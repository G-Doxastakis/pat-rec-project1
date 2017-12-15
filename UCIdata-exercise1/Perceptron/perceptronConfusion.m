function [  ] = perceptronConfusion( OutputTrain,testTrain,OutputValid,testValid )
    OT(1,:) = double(OutputTrain > 0)';
    OT(2,:) = double(OutputTrain < 0)';
    TT(1,:) = double(testTrain > 0)';
    TT(2,:) = double(testTrain < 0)';
    OV(1,:) = double(OutputValid > 0)';
    OV(2,:) = double(OutputValid < 0)';
    TV(1,:) = double(testValid > 0)';
    TV(2,:) = double(testValid < 0)';
    plotconfusion(OT,TT,'Training',OV,TV,'Validation')
end

