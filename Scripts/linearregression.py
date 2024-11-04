import numpy as np

class LinearRegression:
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x: np.ndarray = x
        self.y: np.ndarray = y

        self.intercept: bool = False

        if self.x.ndim == 1:
            self.m: float = 0
            self.b: float = 0
            self.multidim: bool = False
        elif self.x.ndim == 2:
            self.beta: np.ndarray | None = None
            self.multidim: bool = True
        else:
            raise ValueError("x must be a 1D or 2D array")

    def fit(self, **kwargs) -> None:
        self.intercept = 'intercept' in kwargs
        if not self.multidim:
            # m = (mean(x * y) - mean(x) * mean(y)) / (mean(x^2) - mean(x)^2)
            self.m = ((np.mean(self.x * self.y) - np.mean(self.x) * np.mean(self.y)) /
                      (np.mean(self.x ** 2) - np.mean(self.x) ** 2))
            if self.intercept:
                # b = mean(y) - m * mean(x)
                self.b = np.mean(self.y) - self.m * np.mean(self.x)
        else:
            if self.intercept:
                self.beta = np.zeros(self.x.shape[1] + 1)
                x_b = np.c_[np.ones(self.x.shape[0]), self.x]
            else:
                self.beta = np.zeros(self.x.shape[1])
                x_b = self.x
            try:
                self.beta = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ self.y
            except np.linalg.LinAlgError:
                # Iterative computation by successive approximations
                # Learning rate
                alpha = 0.001
                # Convergence tolerance
                tolerance = 1e-6
                # Maximum iterations (num of steps)
                max_iterations = 1000
                for iteration in range(max_iterations):
                    # Compute the gradient
                    if np.isinf(gradient := x_b.T @ (x_b @ self.beta - self.y) ).any():
                        self.beta = np.linalg.pinv(x_b.T @ x_b) @ x_b.T @ self.y
                        break
                    # Clip the gradient: fixing the leght of the numerical gradient
                    np.clip(gradient, -1e10, 1e10)
                    new_beta = self.beta - alpha * gradient
                    # Check for convergence
                    if np.linalg.norm(new_beta - self.beta) < tolerance:
                        break
                        # beta = (x^T * x)^-1 * x^T * y
                    self.beta = new_beta



    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.multidim and x.ndim == 1:
            return self.m * x + self.b
        elif self.multidim and x.ndim == 2:
            if self.intercept:
                return np.c_[np.ones(x.shape[0]), x] @ self.beta
            else:
                return x @ self.beta
        else:
            raise ValueError("x must be a 1D or 2D array. The same used for training.")

