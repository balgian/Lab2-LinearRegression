import os
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from linearregression import LinearRegression


def main() -> None:
    # Task 1: Get data
    path_data: str = os.path.join(os.path.dirname(__file__), "Data")

    turkish_data: pd.DataFrame = pd.read_csv(os.path.join(path_data, "turkish-se-SP500vsMSCI.csv"),
                                             sep=",", header=None)
    mtk_data: pd.DataFrame = pd.read_csv(os.path.join(path_data, "mtcarsdata-4features.csv"), header=0)

    x_turkish_train: list[np.ndarray] = list()
    y_turkish_train: list[np.ndarray] = list()

    x_turkish_test: list[np.ndarray] = list()
    y_turkish_test: list[np.ndarray] = list()

    # Task 2: Fit a linear regression model

    # Split the turkish data into train and test
    for i in range(10):
        turkish_data = turkish_data.sample(n=len(turkish_data))

        x_turkish: np.ndarray = turkish_data[0].values
        y_turkish: np.ndarray = turkish_data[1].values

        x_turkish_train.append(x_turkish[:int(len(x_turkish) * 0.1)+1])
        y_turkish_train.append(y_turkish[:int(len(y_turkish) * 0.1)+1])

        x_turkish_test.append(x_turkish[int(len(x_turkish) * 0.05)+1:])
        y_turkish_test.append(y_turkish[int(len(y_turkish) * 0.05)+1:])

    # Split the mtk data into train and test
    mkt_data = mtk_data.sample(n=len(mtk_data))

    x_mkt: np.ndarray = mkt_data.drop(columns=["Model", " mpg"]).values
    y_mkt: np.ndarray = mkt_data[" mpg"].values

    x_mkt_train: np.ndarray = x_mkt[:int(len(x_mkt) * 0.05)+1]
    y_mkt_train: np.ndarray = y_mkt[:int(len(y_mkt) * 0.05)+1]

    x_mkt_test: np.ndarray = x_mkt[int(len(x_mkt) * 0.05)+1:]
    y_mkt_test: np.ndarray = y_mkt[int(len(y_mkt) * 0.05)+1:]

    # Task 3: Test regression model
    # Test the turkish data and plot
    # Without intercept
    for i in range(10):
        model: LinearRegression = LinearRegression(x_turkish_train[i], y_turkish_train[i])
        model.fit()
        y_hat = model.predict(x_turkish_test[i])

        # Compute the mean squared error: mse = np.mean((y - y_pred) ** 2)
        # Turkish data
        mse_turk = np.mean((x_turkish_test - y_hat) ** 2)
        mse_turk_train = np.mean((x_turkish_train[i] - model.predict(x_turkish_train[i])) ** 2)
        print(f"Turkish data without intercept subset {i + 1}")
        print(f"MSE on test data: {mse_turk}")
        print(f"MSE on train data: {mse_turk_train}\n")

        # Plot
        plt.scatter(x_turkish_test[i], y_turkish_test[i], color=f"C{i}", zorder=1, alpha=0.1, marker=".")
        plt.plot(x_turkish_test[i], y_hat, label=f"Model subset  {i + 1}", color=f"C{i}", alpha=0.9, linewidth=0.5,
                 zorder=2)

    plt.title("Linear Regression Model")
    plt.xlabel("Standard and Poor's 500 return index", fontsize="small")
    plt.ylabel("MSCI Europe index", fontsize="small", labelpad=2)
    plt.grid(True, which="both", linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.25)
    plt.legend(loc="best", fontsize="small", frameon=False)
    plt.show()

    # With intercept
    for i in range(10):
        model: LinearRegression = LinearRegression(x_turkish_train[i], y_turkish_train[i])
        model.fit(intercept=True)
        y_hat = model.predict(x_turkish_test[i])

        # Compute the mean squared error: mse = np.mean((y - y_pred) ** 2)
        # Turkish data
        mse_turk = np.mean((x_turkish_test - y_hat) ** 2)
        mse_turk_train = np.mean((x_turkish_train[i] - model.predict(x_turkish_train[i])) ** 2)
        print(f"Turkish data with intercept subset {i + 1}")
        print(f"MSE on test data: {mse_turk}")
        print(f"MSE on train data: {mse_turk_train}\n")

        plt.scatter(x_turkish_test[i], y_turkish_test[i], color=f"C{i}", zorder=1, alpha=0.1, marker=".")
        plt.plot(x_turkish_test[i], y_hat, label=f"Model subset  {i + 1}", color=f"C{i}", alpha=0.9, linewidth=0.5,
                 zorder=2)

    plt.title("Linear Regression Model with intercept")
    plt.xlabel("Standard and Poor's 500 return index", fontsize="small")
    plt.ylabel("MSCI Europe index", fontsize="small", labelpad=2)
    plt.grid(True, which="both", linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.25)
    plt.legend(loc="best", fontsize="small", frameon=False)
    plt.show()

    ########################################################################
    # Test the mtk data and plot
    # Without intercept
    model: LinearRegression = LinearRegression(x_mkt_train[:, 2], y_mkt_train)
    model.fit()
    y_hat = model.predict(x_mkt_test[:, 2])

    plt.scatter(x_mkt_test[:, 2], y_mkt_test, color=f"C1", zorder=1, alpha=0.4, marker=".")
    plt.plot(x_mkt_test[:, 2], y_hat, color=f"C1", alpha=0.9, linewidth=0.5,
             zorder=2)

    plt.title("Linear Regression Model without intercept")
    plt.xlabel("Weight", fontsize="small")
    plt.ylabel("MPG", fontsize="small", labelpad=2)
    plt.grid(True, which="both", linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.25)
    plt.show()

    ########################################################################
    # Test the mtk data and plot
    # With intercept
    model: LinearRegression = LinearRegression(x_mkt_train[:, 2], y_mkt_train)
    model.fit(intercept=True)
    y_hat = model.predict(x_mkt_test[:, 2])

    plt.scatter(x_mkt_test[:, 2], y_mkt_test, color=f"C1", zorder=1, alpha=0.4, marker=".")
    plt.plot(x_mkt_test[:, 2], y_hat, color=f"C1", alpha=0.9, linewidth=0.5,
             zorder=2)

    plt.title("Linear Regression Model with intercept")
    plt.xlabel("Weight", fontsize="small")
    plt.ylabel("MPG", fontsize="small", labelpad=2)
    plt.grid(True, which="both", linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.25)
    plt.show()

    # Compute the mean squared error: mse = np.mean((y - y_pred) ** 2)
    # Mtk data
    mse_test = np.mean((y_mkt_test - y_hat) ** 2)
    mse_train = np.mean((y_mkt_train - model.predict(x_mkt_train[:, 2])) ** 2)
    print(f"Mtk data, with only mpg and weight, without intercept")
    print(f"MSE on test data: {mse_test}")
    print(f"MSE on train data: {mse_train}\n")

    ########################################################################
    # Without intercept
    model: LinearRegression = LinearRegression(x_mkt_train, y_mkt_train)
    model.fit()
    y_hat = model.predict(x_mkt_test)

    # Extracting the columns from x_mkt_test
    x1 = x_mkt_test[:, 0]
    x2 = x_mkt_test[:, 1]
    x3 = x_mkt_test[:, 2]
    # Create a grid for x1 and x2
    xi = np.linspace(np.min(x1), np.max(x1), 100)
    yi = np.linspace(np.min(x2), np.max(x2), 100)
    xi, yi = np.meshgrid(xi, yi)
    # Interpolate the corresponding x3 values based on the grid
    zi = griddata((x1, x2), x3, (xi, yi), method='linear')
    # Interpolate y_hat values for the grid
    yi_hat = griddata((x1, x2), y_hat, (xi, yi), method='linear')
    # Normalize the interpolated y_hat for coloring
    norm_yi_hat = (yi_hat - np.nanmin(yi_hat)) / (np.nanmax(yi_hat) - np.nanmin(yi_hat))

    # Create a figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, 60)
    # Create a surface plot using the grid
    surf = ax.plot_surface(xi, yi, zi, facecolors=cm.coolwarm(norm_yi_hat), linewidth=0.2, antialiased=True, alpha=0.5)
    # Create a color bar for the surface
    mappable = cm.ScalarMappable(cmap=cm.coolwarm)
    mappable.set_array(norm_yi_hat)  # Set the array for the color mapping
    mappable.set_clim(0, 1)  # Set the color limits
    fig.colorbar(mappable, ax=ax, label='Normalized Y_hat Values')
    # Set labels for axes
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    # Show the plot
    plt.title('3D Surface Plot Colored by Y_hat')
    plt.show()

    # Compute the mean squared error: mse = np.mean((y - y_pred) ** 2)
    # Mtk data
    mse_test = np.mean((y_mkt_test - y_hat) ** 2)
    mse_train = np.mean((y_mkt_train - model.predict(x_mkt_train)) ** 2)
    print(f"Mtk data with intercept")
    print(f"MSE on test data: {mse_test}")
    print(f"MSE on train data: {mse_train}\n")


    ########################################################################
    # With intercept
    model: LinearRegression = LinearRegression(x_mkt_train, y_mkt_train)
    model.fit(intercept=True)
    y_hat = model.predict(x_mkt_test)

    # Extracting the columns from x_mkt_test
    x1 = x_mkt_test[:, 0]
    x2 = x_mkt_test[:, 1]
    x3 = x_mkt_test[:, 2]
    # Create a grid for x1 and x2
    xi = np.linspace(np.min(x1), np.max(x1), 100)
    yi = np.linspace(np.min(x2), np.max(x2), 100)
    xi, yi = np.meshgrid(xi, yi)
    # Interpolate the corresponding x3 values based on the grid
    zi = griddata((x1, x2), x3, (xi, yi), method='linear')
    # Interpolate y_hat values for the grid
    yi_hat = griddata((x1, x2), y_hat, (xi, yi), method='linear')
    # Normalize the interpolated y_hat for coloring
    norm_yi_hat = (yi_hat - np.nanmin(yi_hat)) / (np.nanmax(yi_hat) - np.nanmin(yi_hat))

    # Create a figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, 60)
    # Create a surface plot using the grid
    surf = ax.plot_surface(xi, yi, zi, facecolors=cm.coolwarm(norm_yi_hat), linewidth=0.2, antialiased=True, alpha=0.5)
    # Create a color bar for the surface
    mappable = cm.ScalarMappable(cmap=cm.coolwarm)
    mappable.set_array(norm_yi_hat)  # Set the array for the color mapping
    mappable.set_clim(0, 1)  # Set the color limits
    fig.colorbar(mappable, ax=ax, label='Normalized Y_hat Values')
    # Set labels for axes
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    # Show the plot
    plt.title('3D Surface Plot Colored by Y_hat')
    plt.show()

    # Compute the mean squared error: mse = np.mean((y - y_pred) ** 2)
    # Mtk data
    mse_test = np.mean((y_mkt_test - y_hat) ** 2)
    mse_train = np.mean((y_mkt_train - model.predict(x_mkt_train)) ** 2)
    print(f"Mtk data with intercept")
    print(f"MSE on test data: {mse_test}")
    print(f"MSE on train data: {mse_train}\n")


if __name__ == "__main__":
    main()
