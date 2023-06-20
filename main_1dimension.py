import numpy as np
from numpy import argmax, arange, asarray
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from numpy import asarray, vstack
import pandas as pd

# objective function
def objective(x, noise=0.1):
    return (x-0.7)**2

# grid-based sample of the domain [0,1]
X = arange(0, 1, 0.01)
# sample the domain without noise
y = [objective(x, 0) for x in X]
ynoise = [objective(x) for x in X]

# 최적점 x,y 찾기
ix = argmax(y)
print('Optima: x=%.3f, y=%.3f' % (X[ix], y[ix]))

def surrogate(model, X):
	return model.predict(X, return_std=True)

# Expected Improvement(EI)의 probability of improvement(PI)를 계산하는 함수
def acquisition(X, Xsamples, model):
	# 현재 데이터셋에서 가장 큰 yhat = f(xhat)를 찾는다
	yhat, _ = surrogate(model, X)
	xhat = max(yhat)
	# surrogate 함수를 이용하여 Sample들의 mean, std를 계산한다.
	mu, std = surrogate(model, Xsamples)
	# PI를 계산한다.
	probs = norm.cdf((mu - xhat) / (std+1E-9))
	return probs

# random search로 acquisition function을 계산하여 최적값을 선택하는 함수
def opt_acquisition(X, y, model):
	# random search 방식으로 random sample을 생성한다.
	Xsamples = random(100)
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	# 샘플별 acquisition 함수를 통해 스코어를 계산한다.
	scores = acquisition(X, Xsamples, model)
	# 가장 높은 스코어를 뽑는다.
	ix = argmax(scores)
	return Xsamples[ix, 0]

X = random(100)
y = asarray([objective(x) for x in X])
# reshape into rows and cols
X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)
# surrogate functino에 사용할 Gaussian Process Regressor 모델
model = GaussianProcessRegressor()
# 지금까지 Sample로 학습한다.
model.fit(X, y)

# Bayesian Optimization을 수행
for i in range(100):
	# 다음 샘플을 얻는다
	x_sample = opt_acquisition(X, y, model)
	# 목적함수를 거친 결과를 얻는다
	y_sample = objective(x_sample)
	# summarize the finding
	est, _ = surrogate(model, [[x_sample]])
	print('>x=%.3f, f()=%3f, actual=%.3f' % (x_sample, est, y_sample))
	# 새로운 x_sample, y_sample을 지금까지의 데이터셋에 추가한다
	X = vstack((X, [[x_sample]]))
	y = vstack((y, [[y_sample]]))
	# update surrogate the model
	model.fit(X, y)

# 최적값을 출력한다.
ix = argmax(y)
print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))

df = pd.DataFrame(np.hstack((X,y)))
df.to_csv('sample_dim1.csv', index=False)
