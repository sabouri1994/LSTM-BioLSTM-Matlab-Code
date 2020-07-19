clc
clear all
%% Data 
load('train and test pulse.mat')
%% Normalaize Data
mu = mean(Data_Train,2);
sig = std(Data_Train,0,2);
trainingFeatures=zeros(size(Data_Train));
for i=1:size(Data_Train,1)
    for index = 1:size(Data_Train,2)
      trainingFeatures(i,index) =((Data_Train(i,index) - mu(i))./sig(i)).';
    end
end
a=1;
b=3;
c=21;
XTrain=trainingFeatures(a:b,:);
YTrain=trainingFeatures(c,:);

mu1 = mean(Data_Test_pulse,2);
sig1 = std(Data_Test_pulse,0,2);
trainingFeatures=zeros(size(Data_Test_pulse));
for i=1:size(Data_Test_pulse,1)
    for index = 1:size(Data_Test_pulse,2)
      trainingFeatures(i,index) =((Data_Test_pulse(i,index) - mu1(i))./sig1(i)).';
    end
end

Xtest=trainingFeatures(a:b,:);
Ytest=trainingFeatures(c,:);

%% Data Validaton
idx = randperm(size(XTrain,2),1000);
XValidation = XTrain(1:20,idx);
YValidation = YTrain(idx);
%% Made LSTM Network

inputSize = (b-a)+1;
outputSize= size(YTrain,1);
numResponses = 50;
layers = [ ...
    sequenceInputLayer(inputSize,'Name','input1')
    lstmLayer(100,'Name','input2')
    bilstmLayer(100,'Name','input3')
    fullyConnectedLayer(outputSize,'Name','input11')
    %fullyConnectedLayer(outputSize)
    regressionLayer]
%gruLayer(10)

maxEpochs = 1000
miniBatchSize = 50;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'Shuffle','never',...
    'Plots','training-progress',...
    'Verbose',1,...
    'VerboseFrequency', 5)

%% Train Network
net = trainNetwork(XTrain,YTrain,layers,options);

%% Test Network
[net,score] = predictAndUpdateState(net,XTrain);

%%
YPred = predict(Net,XTest,'MiniBatchSize',1);
%%
YPred = sig1(c).*YPred + mu1(c);
% rmse = sqrt(mean((YPred-Ytest).^2,2))
