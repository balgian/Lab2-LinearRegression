import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearregression import LinearRegression

# Questo dovrebbe essere il mio fork

def main() -> None:
    # Task 1: Get data
    path_data: str = os.path.join(os.path.dirname(__file__), "Data")

    turkish_data: pd.DataFrame = pd.read_csv(os.path.join(path_data, "turkish-se-SP500vsMSCI.csv"), header=None)
    mtk_data: pd.DataFrame = pd.read_csv(os.path.join(path_data, "mtcarsdata-4features.csv"), header=0)

    x_turkish_train: list[np.ndarray] = list()
    y_turkish_train: list[np.ndarray] = list()

    x_turkish_test: list[np.ndarray] = list()
    y_turkish_test: list[np.ndarray] = list()

    # Task 2: Fit a linear regression model
    for i in range(10):
        turkish_data = turkish_data.sample(n=len(turkish_data))
        mkt_data = mtk_data.sample(n=len(mtk_data))

        x_turkish: np.ndarray = turkish_data[0].values
        y_turkish: np.ndarray = turkish_data[1].values

        x_turkish_train.append(x_turkish[:int(len(x_turkish) * 0.1)])
        y_turkish_train.append(y_turkish[:int(len(y_turkish) * 0.1)])

        x_turkish_test.append(x_turkish[int(len(x_turkish) * 0.9):])
        y_turkish_test.append(y_turkish[int(len(y_turkish) * 0.9):])

    # Task 3: Test regression model
    for i in range(10):
        model: LinearRegression = LinearRegression(x_turkish_train[i], y_turkish_train[i])
        model.fit()
        y_hat = model.predict(x_turkish_test[i])

        plt.scatter(x_turkish_test[i], y_turkish_test[i], color=f"C{i}", zorder=1, alpha=0.1, marker=".")
        plt.plot(x_turkish_test[i], y_hat, label=f"Model subset  {i + 1}", color=f"C{i}", alpha=0.9, linewidth=0.5,
                 zorder=2)

    plt.title("Linear Regression Model")
    plt.xlabel("Standard and Poor's 500 return index", fontsize="small")
    plt.ylabel("MSCI Europe index", fontsize="small", labelpad=2)
    plt.grid(True, which="both", linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.25)
    plt.legend(loc="best", fontsize="small", frameon=False)
    plt.show()

    for i in range(10):
        model: LinearRegression = LinearRegression(x_turkish_train[i], y_turkish_train[i])
        model.fit(intercept=True)
        y_hat = model.predict(x_turkish_test[i])

        plt.scatter(x_turkish_test[i], y_turkish_test[i], color=f"C{i}", zorder=1, alpha=0.1, marker=".")
        plt.plot(x_turkish_test[i], y_hat, label=f"Model subset  {i + 1}", color=f"C{i}", alpha=0.9, linewidth=0.5,
                 zorder=2)

    plt.title("Linear Regression Model with intercept")
    plt.xlabel("Standard and Poor's 500 return index", fontsize="small")
    plt.ylabel("MSCI Europe index", fontsize="small", labelpad=2)
    plt.grid(True, which="both", linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.25)
    plt.legend(loc="best", fontsize="small", frameon=False)
    plt.show()


if __name__ == "__main__":
    main()
