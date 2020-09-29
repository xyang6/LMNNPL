function X = data_preprocess(X)

N      = size(X,1);
%mean centering
X_mean = mean(X,1);
X      = X - repmat(X_mean,N,1);clear X_mean
%standardisation
X_sd   = std(X,1);
X      = X ./ repmat(X_sd,N,1);clear X_sd
%L2 normalisation
X      = X ./ vecnorm(X,2,2);

end