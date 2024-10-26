import numpy as np


class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.m = 0
        self.b = 0
        ...

    def fit(self, **kwargs):
        # m = (mean(x * y) - mean(x) * mean(y)) / (mean(x^2) - mean(x)^2)
        self.m = ((np.mean(self.x * self.y) - np.mean(self.x) * np.mean(self.y)) /
                  (np.mean(self.x ** 2) - np.mean(self.x) ** 2))
        if 'intercept' in kwargs:
            # b = mean(y) - m * mean(x)
            self.b = np.mean(self.y) - self.m * np.mean(self.x)

    def predict(self, x):
        return self.m * x + self.b

    def __str__(self):
        return f"y = {self.m}x + {self.b}"