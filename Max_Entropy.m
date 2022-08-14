% This code uses Maximum Entropy to evaluate the dependence of the price
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
% Train the Max Entropy Model
Entropy_Model = mnrfit(F(idx(1:train),1:p-1),Y(idx(1:train)));  

% Model prediction
Y_predict= zeros(test,1);
for i = train+1:N
    if(Entropy_Model(1) + Entropy_Model(2:p)'*F(idx(i),1:p-1)' > 0) %decision boundary
        Y_predict((i-train),1) = 1;
    else
        Y_predict((i-train),1) = 2;
    end
end

% Accuracy calculation
Accuracy = 100*sum((Y_predict == Y(idx(train+1:end))))/(test)    


