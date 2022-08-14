% This code uses Support Vector Machine to evaluate the dependence of the price
% change on the sentiment depicted by the tweets on Bitcoins 

load 'data.mat'
rng(5)

% Calculate the size of the dataset(preprocessed features)
[N,p] = size(data);                                  
Y = zeros(N,1); 
F = table2array(data);
for i= 1:N
    % Assign the class labels based on the bitcoin price change
    if F(i,p) < 0                                        
        Y(i) = 1;
    else
        Y(i) = -1;
    end
end
% Divide the dataset randomly into testing and training data
idx=randperm(numel(Y));        
train = floor(0.7*N);
test = N - train;

% Train the SVM Model
SVM_Model = fitcsvm(F(idx(1:train),1:p-1),Y(idx(1:train)),'KernelScale','auto');

% Model prediction
Y_predict = predict(SVM_Model,F(train+1:end,1:p-1));

% Accuracy calculation
Accuracy = 100*sum((Y_predict.*Y(idx(train+1:end))) > 0)/(test)
