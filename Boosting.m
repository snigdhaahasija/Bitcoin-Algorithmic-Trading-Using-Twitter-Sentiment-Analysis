% This code uses Boosting Algorithms to evaluate the dependence of the price
% change on the sentiment depicted by the tweets on Bitcoins 

load 'data.mat'
rng(5)

% Calculate the size of the dataset(preprocessed features)
[N,p] = size(data);                                  
Y = zeros(N,1); 
F = table2array(data);
for i= 1:N
    % Assign the class labels based on the bitcoin price change
    if F(i,p) > 0                                        
        Y(i) = 1;
    else
        Y(i) = 2;
    end
end
% Divide the dataset randomly into testing and training data
idx=randperm(numel(Y));        
train = floor(0.7*N);
test = N - train;

% Train the AdaBoost Model
AdaBoost_Model = fitcensemble(F(idx(1:train),1:p-1),Y(idx(1:train)),'Method','AdaBoostM1');  

% Model prediction
Y_predict = predict(AdaBoost_Model,F(train+1:end,1:p-1));

% Accuracy calculation
AdaBoost_Accuracy = 100*sum((Y_predict == Y(idx(train+1:end))))/(test)

% Train the Bag Model
Bag_Model = fitcensemble(F(idx(1:train),1:p-1),Y(idx(1:train)),'Method','Bag');  

% Model prediction
Y_predict = predict(Bag_Model,F(train+1:end,1:p-1));

% Accuracy calculation
Bag_Accuracy = 100*sum((Y_predict == Y(idx(train+1:end))))/(test) 

% Train the GentleBoost Model
GentleBoost_Model = fitcensemble(F(idx(1:train),1:p-1),Y(idx(1:train)),'Method','GentleBoost');  

% Model prediction
Y_predict = predict(GentleBoost_Model,F(train+1:end,1:p-1));

% Accuracy calculation
GentleBoost_Accuracy = 100*sum((Y_predict == Y(idx(train+1:end))))/(test) 

% Train the LogitBoost Model
LogitBoost_Model = fitcensemble(F(idx(1:train),1:p-1),Y(idx(1:train)),'Method','LogitBoost');  

% Model prediction
Y_predict = predict(LogitBoost_Model,F(train+1:end,1:p-1));

% Accuracy calculation
LogitBoost_Accuracy = 100*sum((Y_predict == Y(idx(train+1:end))))/(test)
