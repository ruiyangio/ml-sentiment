import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score
from modelbase import ModelBase

class LogisticRegression(object):
    def __init__(self, max_iteration = 10000, learning_rate = 3e-5, add_intercept = True):
        self.max_iteration = max_iteration
        self.learning_rate = learning_rate
        self.add_intercept = add_intercept
        self.w = None

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def cost(self, y, y_pred):
        n_sample = y.shape[0]
        cross_entropy = -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)
        return cross_entropy.sum() / n_sample

    def fit(self, x, y):
        if self.add_intercept:
            intercept = np.ones((x.shape[0], 1))
            x = sparse.hstack((intercept, x))

        self.w = np.zeros(x.shape[1])
        for i in range(self.max_iteration):
            scores = x.dot(self.w)
            y_pred = self.sigmoid(scores)
            error = y - y_pred
            gradient = x.T.dot(error)
            self.w += self.learning_rate * gradient
        print("train complete")

    def predict(self, x):
        if self.add_intercept:
            intercept = np.ones((x.shape[0], 1))
            x = sparse.hstack((intercept, x))

        scores = x.dot(self.w)
        return np.round(self.sigmoid(scores))

class MyLogisticRegression(ModelBase):
    def __init__(self):
        ModelBase.__init__(self)
        self.model = LogisticRegression()