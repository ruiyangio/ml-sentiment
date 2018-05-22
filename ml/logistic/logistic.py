import numpy as np
import sklearn

class LogisticRegression(object):
    def __init__(self, max_iteration = 2000, learning_rate = 5e-5, add_intercept = False):
        self.max_iteration = max_iteration
        self.learning_rate = learning_rate
        self.add_intercept = add_intercept
        self.w = None

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def train(self, x, y):
        x = np.array(x.toarray())
        if self.add_intercept:
            intercept = np.ones((x.shape[0], 1))
            x = np.hstack((intercept, x))    
        self.w = np.zeros(x.shape[1])

        for i in range(self.max_iteration):
            weights = np.dot(x, self.w)
            res = self.sigmoid(weights)

            error = y - res
            # gradient descent
            grad = np.dot(x.T, error)
            self.w += self.learning_rate * grad
        print("train complete")

    def test(self, x, y):
        x = np.array(x.toarray())
        scores = np.dot(x, self.w)
        y_pred = np.round(self.sigmoid(scores))
        return sklearn.metrics.accuracy_score(y, y_pred)
