import marimo

__generated_with = "0.9.31"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    return mo, np, plt, sns


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Линейная регрессия

        У нас есть некоторые наблюдения $\{x_i, y_i\}$, мы хотим: предсказать $y$ для новых $x$, которые у нас появятся в ходе дальнейшей эксплуатации изделия (или просто использования модели).

        Простейший вариант подбора зависимости $y(x)$ является линейная зависимость в виде:
        $$\overrightarrow{y} =  X \overrightarrow{w} + b$$
        где $\overrightarrow{w}$ - вектор параметров модели, $b$ - базовое смещение или среднее модели (intercept).

        Для упрощения без потери общности модеи параметр $b$ добавляют в $\overrightarrow{w}$ путем введение фиктивного столбца признаков, который всегда равен единице:

        $$\tilde{X} = [1 \quad X]$$

        Подбор параметров линейной регрессии производится методом наименьших квадратов. Определяется остаток (ошибка применения модели):

        $e_i = y_i - \hat{y_i}$
        где $y_i$ - реальное значение из обучающей выборки, $\hat{y_i}$ - предсказанное в соответсвии с моделью.

        Метод наименьших квадратов подбирает параметры модели таким образом, чтобы достигался минимум суммы квадратов отклонений:

        $$\overrightarrow{w} = \argmin \sum (y_i - \tilde X \overrightarrow{w} )^2$$

        Для квадратичной функции минимум достигается в точке, где производная функция равна 0, поэтому оптимальными параметрами модели будут такие, для которых выполняется равенство:
        $$\nabla S(\overrightarrow{w}) = 2\tilde X^T(y-\tilde X\overrightarrow{w}) = 0$$
        Условие минимума достигается при :
        $$\tilde X^T \tilde X \overrightarrow{w}=\tilde X^T y$$

        Полученная система уравенний решается любым известным методом, например, методом Гаусса.
        """
    )
    return


@app.cell(hide_code=True)
def __(np):
    def make_linear_regression_dataset(
        n: int = 200,
        p: int = 1,
        w: np.ndarray | None = None,
        b: float = 0.5,
        noise_sigma: float = 1.0,
        x_range: tuple[float, float] = (-2.0, 2.0),
        seed: int | None = 0,
    ):
        """
        Generate synthetic linear regression dataset: y = b + X @ w + eps.

        Parameters
        ----------
        n : int
            Number of samples.
        p : int
            Number of features (1 or 2 recommended).
        w : np.ndarray | None
            True weights shape (p,). If None, uses defaults for p=1 or p=2.
        b : float
            True intercept.
        noise_sigma : float
            Std dev of Gaussian noise eps ~ N(0, noise_sigma^2).
        x_range : (float, float)
            Uniform range for features.
        seed : int | None
            RNG seed for reproducibility.

        Returns
        -------
        X : np.ndarray, shape (n, p)
        y : np.ndarray, shape (n,)
        params : dict
            Dictionary with ground-truth parameters {"b": b, "w": w, "noise_sigma": noise_sigma}.
        """
        if p < 1:
            raise ValueError("p must be >= 1 (use p=1 or p=2 for this exercise).")

        rng = np.random.default_rng(seed)

        lo, hi = x_range
        X = rng.uniform(lo, hi, size=(n, p))

        if w is None:
            if p == 1:
                w = np.array([2.0], dtype=float)
            elif p == 2:
                w = np.array([1.5, -0.7], dtype=float)
            else:
                # deterministic but nontrivial default
                w = rng.normal(0.0, 1.0, size=p)

        w = np.asarray(w, dtype=float)
        if w.shape != (p,):
            raise ValueError(f"w must have shape ({p},), got {w.shape}.")

        eps = rng.normal(0.0, noise_sigma, size=n)
        y = b + X @ w + eps

        params = {"b": float(b), "w": w.copy(), "noise_sigma": float(noise_sigma)}
        return X, y, params
    return (make_linear_regression_dataset,)


@app.cell
def __(make_linear_regression_dataset):
    x, y, _ = make_linear_regression_dataset(n=100, p=1, w=[2], b=0.5, noise_sigma=1)
    return x, y


@app.cell
def __(np, x):
    X_tild = np.hstack((np.ones((x.shape[0], 1)), x))
    return (X_tild,)


@app.cell
def __(X_tild, np, y):
    p_est = np.linalg.solve(X_tild.T @ X_tild, X_tild.T @ y)
    return (p_est,)


@app.cell
def __(p_est):
    p_est
    return


@app.cell
def __(X_tild, p_est):
    y_est = X_tild @ p_est
    return (y_est,)


@app.cell(hide_code=True)
def __(np, plt):
    def plot_linear_fit(X, y, theta, *, ax=None, n_grid=200, grid=30):
        """
        Universal plot for linear regression fit:
          - 1 feature: 2D scatter + fitted line
          - 2 features: 3D scatter + fitted plane

        Model:
          y_hat = b + X @ w, where
          theta = [b, w1]         for p=1
          theta = [b, w1, w2]     for p=2

        Parameters
        ----------
        X : array-like
            Shape (n,) or (n,1) for 1D; shape (n,2) for 2D.
        y : array-like
            Shape (n,)
        theta : array-like
            Shape (2,) for 1D or (3,) for 2D.
        ax : matplotlib axis or None
            If None, creates a new axis (2D for p=1, 3D for p=2).
        n_grid : int
            Number of points for line (p=1).
        grid : int
            Grid resolution for plane (p=2).

        Returns
        -------
        ax : matplotlib axis
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        theta = np.asarray(theta).ravel()

        # Normalize X shape
        if X.ndim == 1:
            Xn = X.reshape(-1, 1)
        elif X.ndim == 2:
            Xn = X
        else:
            raise ValueError("X must be 1D (n,) or 2D (n,p).")

        if Xn.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same n. Got {Xn.shape[0]} and {y.shape[0]}.")

        p = Xn.shape[1]
        if p not in (1, 2):
            raise ValueError("This plotting helper supports only p=1 or p=2 features.")

        if theta.shape != (p + 1,):
            raise ValueError(f"theta must have shape ({p+1},) = [b, w...], got {theta.shape}.")

        b = theta[0]
        w = theta[1:]

        if p == 1:
            x = Xn[:, 0]

            if ax is None:
                fig, ax = plt.subplots()

            ax.scatter(x, y)

            xg = np.linspace(x.min(), x.max(), n_grid)
            yg = b + w[0] * xg
            ax.plot(xg, yg)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Линейная регрессия: точки + подобранная прямая")
            return ax

        # p == 2
        x1 = Xn[:, 0]
        x2 = Xn[:, 1]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        ax.scatter(x1, x2, y)

        x1g = np.linspace(x1.min(), x1.max(), grid)
        x2g = np.linspace(x2.min(), x2.max(), grid)
        X1g, X2g = np.meshgrid(x1g, x2g)
        Yg = b + w[0] * X1g + w[1] * X2g

        ax.plot_surface(X1g, X2g, Yg, alpha=0.35)

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        ax.set_title("Линейная регрессия: точки + подобранная плоскость")
        return ax
    return (plot_linear_fit,)


@app.cell
def __(p_est, plot_linear_fit, x, y):
    plot_linear_fit(x, y, p_est)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
