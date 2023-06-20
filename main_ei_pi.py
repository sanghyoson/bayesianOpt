import numpy as np
from numpy import argmax, arange, asarray
from numpy.random import random
from numpy import asarray, vstack
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd

# objective function
def objective(x, noise=0):
    noise = np.random.normal(loc=0, scale=noise)
    return (x[0] - 0.7)**2 + (x[1] - 0.3)**2 + noise

# Define the surrogate model
def surrogate(model, X):
    return model.predict(X, return_std=True)

# Acquisition function
def acquisition_pi(X, Xsamples, model):
    yhat, _ = surrogate(model, X)
    xhat = np.max(yhat)
    mu, std = surrogate(model, Xsamples)
    z = (mu - xhat) / (std + 1E-9)
    pi = norm.cdf(z)
    return pi

def opt_acquisition_pi(X, y, model):
    Xsamples = np.random.rand(100, X.shape[1])
    scores = acquisition_pi(X, Xsamples, model)
    ix = np.argmax(scores)
    return Xsamples[ix]

# def acquisition_ei(X, Xsamples, model):
#     yhat, _ = surrogate(model, X)
#     xhat = np.max(yhat)
#     mu, std = surrogate(model, Xsamples)
#     ei = norm.cdf((mu - xhat) / (std+1E-9))
#     return ei
#
# # Optimal acquisition function
# def opt_acquisition_ei(X, y, model):
#     Xsamples = np.random.rand(100, X.shape[1])
#     scores = acquisition_ei(X, Xsamples, model)
#     ix = np.argmax(scores)
#     return Xsamples[ix]

# Randomly generate initial data
X = np.random.rand(100, 2)
y = np.array([objective(x) for x in X])

# reshape y
y = y.reshape(len(y), 1)

# define the model
model = GaussianProcessRegressor()

# fit the model
model.fit(X, y)

# perform the optimization process
for i in range(30):
    # select the next point to sample
    x_sample = opt_acquisition_pi(X, y, model)
    # sample the objective function
    y_sample = objective(x_sample)
    # summarize the finding
    est, _ = surrogate(model, [x_sample])
    # print('>x=%.3f, f()=%3f, current=%.3f' % (x_sample, est, y_sample))
    # print(f'{x_sample}, {est}, {y_sample}')
    # add the data to the dataset
    X = np.vstack((X, [x_sample]))
    y = np.vstack((y, [[y_sample]]))
    # update the model
    model.fit(X, y)

# best result
ix = np.argmax(y)
# print('Best Result (PI): x1=%.3f, x2=%.3f, y=%.3f' % (X[ix,0], X[ix,1], y[ix]))
# print(f'X : {X} \n y : {y}')
df = pd.DataFrame(np.hstack((X,y)))

df.to_csv('sample.csv', index=False)
