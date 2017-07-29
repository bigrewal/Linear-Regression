import numpy as np

def normalise(X):
    m,n = X.shape
    X_norm = np.zeros((m,n))
    feature_means = np.zeros(n)
    feature_sd = np.zeros(n)

    for i in range(n):
        feature = X[:,i]
        mu = np.sum(feature)/m
        sd = np.std(feature)

        feature_means[i] = mu;
        feature_sd[i] = sd;

        feature = (feature - mu)/sd;

        X_norm[:, i] = feature;

    return X_norm,feature_means,feature_sd