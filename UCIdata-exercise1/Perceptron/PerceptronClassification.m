clear;
close all;
load('Data.mat');
[neuron,epochs]=trainPerceptron(InputsTrain,OutputTrain,0.1);
testTrain=testPerceptron(InputsTrain,neuron);
testValid=testPerceptron(InputsValid,neuron);
 
%Plots
figure;
perceptronConfusion( OutputTrain,testTrain,OutputValid,testValid );

figure;
subplot(1,2,1); perceptronVisualization(InputsTrain,OutputTrain,neuron); title('Training');
subplot(1,2,2); perceptronVisualization(InputsValid,OutputValid,neuron); title('Validation');

% %Optimal training rate
% figure;
% rates=linspace(0.001,1,1000);
% for i=1:length(rates)
%     epochs(i)=0;
%     for j=1:10
%         [W,epoch]=trainPerceptron(InputsTrain,OutputTrain,rates(i));
%         epochs(i)= epochs(i)+epoch;
%     end
%     epochs(i)= epochs(i)/10.0;
% end
% plot(rates,epochs);ylabel('Epochs');xlabel('Learning Rate');