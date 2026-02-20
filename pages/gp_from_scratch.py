import marimo

__generated_with = "0.9.31"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Регрессия на основе Гауссовских процессов

        ## Постановка задачи

        У нас есть некоторые наблюдения $\{x_i, y_i\}$, мы хотим:

        - предсказать $y$ для новых $x$
        - получить оценку неопределенности в каждой новой точке $x$

        В отличии от классических алгоритмов регрессии (линейная, полиномиальная) регрессия на основе гауссовского процесса дает не только оценку для $y$ в новых точках, но и оценку дисперсии в этих точках, что имеет очень полезные приложения, например, для задач оптимизации.

        Гауссовский процесс это распределение над функциями. То есть реализацией гауссовского процесса является не случайная величина, а случайная функция

        $$f(x) \sim \mathcal{GP}(m(x),\, k(x,x'))$$

        где $m(x)$ - математическое ожиданиие, а $k(x,x')$ - ковариционная функция ядра, которая задает связь ("похожесть") между двумя точками

        Задавая функцию ядра мы утверждаем (логичное предположение), что если две точки $x$ и $x'$ похожи между собой, то и значения функции $f(x)$ и $f(x')$ должны быть близки

        При этом наблюдения с шумом выглядят как:

        $$y_i = f(x_i) + \varepsilon_i,\quad \varepsilon_i\sim \mathcal{N}(0,\sigma_n^2)$$

        Самый распространенный вид функции ядра:

        $$k(x, x') = \sigma_f^2 \exp\left(\frac12 \frac{(x - x')^2}{l^2}\right)$$

        где $\sigma_f$ - амлитуда корреляции, $l$ - масштаб корреляции

        Параметры $\sigma_f$, $l$ и $\sigma_n$ являются гиперпараметрами модели и могут быть оптимизированы для достижения лучшего качества по критерию максимального правдоподобия

        ## Модель

        Запишем совместное нормальное распределение для обучающих и тестовых точек

        $$\begin{bmatrix} f \\ f_* \end{bmatrix}\sim \mathcal{N}\left(\begin{bmatrix} m \\ m_* \end{bmatrix},\begin{bmatrix} K & K_* \\ K_*^\top & K_{**}\end{bmatrix}\right)$$

        Обозначения:
        - $f$ - $f(X)$ значения функции на обучающей выборке
        - $f_*$ - $f(X_*)$ значения функции на тестовой выборке
        - $K$ - матрица ковариаций обучающих данных
        - $K_*$ матрица ковариаций тестовых и обучающих данных
        - $K_{**}$ матрица ковариаций тестовых данных

        Поскольку наблюдения $y = f(x) + \varepsilon,\quad \varepsilon_i\sim \mathcal{N}(0,\sigma_n^2)$, то $y \sim \mathcal{N}(m,\, K+\sigma_n^2 I)=\mathcal{N}(m, K_y)$ и соответсвенно:

        $$\begin{bmatrix} y \\ f_* \end{bmatrix}\sim \mathcal{N}\left(\begin{bmatrix} m \\ m_* \end{bmatrix},\begin{bmatrix} K_y & K_* \\ K_*^\top & K_{**}\end{bmatrix}\right)$$

        Применяя формулу для условного среднего совместного нормального распределения получаем прогноз для тестовых данных:

        $$\mu_* = m_* + K_*^\top K_y^{-1}(y-m)$$

        Если данные нормированы ($\frac{x - \mu_x}{ \sigma_x}$), то $m_*$ и $m$ нулевые и выражение выше можно переписать как:

        $$\mu_* = K_*^\top K_y^{-1}y$$

        Обратная матрица $K_y^{-1}$ явным образом не вычисляется. Для нахождения $\mu_*$ сначала, используя разложение Холецкого, матрица $K_y^{-1}$ представляется в виде $K_y^{-1} = L L^T$, где $L$ нижнетреугодьная матрица. После этого используя алгоритм прямой подстановки решается последовательно две системы уравнений с нижней треугольной матрицей $L^T\alpha = y$ и $L \beta = \alpha$. После этого становится возможным вычисление $\mu_* = K_*^\top \beta$

        Для оценки неопределенности после вычисления всех необходимых параметров производится вычисление дисперсии в каждой тестовой точке:

        $$L V = K_*$$

        $$\Sigma_* = K_{**} - V^T V$$

        ## Интерпретация предсказания и неопределённости

        Предиктивное распределение для латентной функции в тестовых точках имеет вид:

        $$
        f_* \mid X,y,X_* \sim \mathcal{N}(\mu_*,, \Sigma_*)
        $$

        ### 1) Что означает ($\mu_*$)

        Формула

        $$
        \mu_* = m_* + K_*^\top K_y^{-1}(y-m)
        $$

        показывает, что средний прогноз — это **линейная комбинация наблюдений** (y), где веса определяются:

        * «похожестью» тестовой точки на обучающие (через ($K_*$)),
        * взаимной коррелированностью обучающих точек и уровнем шума (через ($K_y^{-1}$)).

        Интуитивно: если ($x_*$) близка к некоторым ($x_i$), то соответствующие ($y_i$) получают большой вес; если точки сильно шумные (большая ($\sigma_n$)), веса «размазываются» и модель меньше «прилипает» к данным.

        ### 2) Что означает ($\Sigma_*$) и почему она убывает возле данных

        Дисперсия определяется выражением

        $$
        \Sigma_* = K_{**} - K_*^\top K_y^{-1}K_*
        $$

        Здесь ($K_{**}$) — **априорная** неопределённость в тестовых точках (до учёта данных), а вычитаемое слагаемое — **сколько неопределённости “снимают” наблюдения**.

        Отсюда сразу следует важная прикладная картина:

        * **рядом с обучающими точками** (где ($K_*$) велико) второе слагаемое большое ⇒ ($\Sigma_*$) уменьшается;
        * **вдали от данных** ($K_*\approx 0$) ⇒ ($\Sigma_* \approx K_{**}$), то есть неопределённость возвращается к априорной.

        ### 3) Разница между прогнозом функции и прогнозом наблюдения

        Важно различать:

        * прогноз **латентной функции** ($f_*$): дисперсия ($\Sigma_*$);
        * прогноз **наблюдения** ($y_* = f_* + \varepsilon$): дисперсия больше на шум:

        $$
          \mathrm{Var}(y_*) = \Sigma_* + \sigma_n^2 I
        $$

        Практически: интервалы для ($y_*$) всегда шире, чем для ($f_*$).

        ### 4) Как гиперпараметры меняют поведение (на уровне “если–то”)

        * Увеличение ($l$): корреляция «дальнодействующая» ⇒ функция более гладкая, влияние точки распространяется дальше.
        * Увеличение ($\sigma_f$): растёт априорная амплитуда колебаний ⇒ растут масштабы ($\mu$) и неопределённости.
        * Увеличение ($\sigma_n$): данных «меньше видно» ⇒ ($\mu_*$) меньше следует за точками, интервалы расширяются.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sn
    import scipy as sc
    import pandas as pd
    import marimo as mo
    return mo, np, pd, plt, sc, sn


@app.cell(hide_code=True)
def _(np, pd):
    def f_true(x: np.ndarray) -> np.ndarray:
        # True latent function
        return np.sin(1.5 * x) + 0.3 * np.cos(4.0 * x)

    def make_1d_dataset(
        n_train: int = 50,
        x_min: float = -3.0,
        x_max: float = 3.0,
        sigma_n: float = 0.08,   # noise std (adjust as needed)
        seed: int = 42,
        n_test: int = 200,
    ) -> None:
        rng = np.random.default_rng(seed)

        # Train inputs: sorted for nicer plotting, but not required
        x_train = rng.uniform(x_min, x_max, size=n_train)
        x_train.sort()

        y_clean = f_true(x_train)
        y_train = y_clean + rng.normal(0.0, sigma_n, size=n_train)

        train_df = pd.DataFrame({"x": x_train, "y": y_train})

        # Test grid (dense): for plotting mean/CI vs ground truth
        x_test = np.linspace(x_min, x_max, n_test)
        test_df = pd.DataFrame({"x": x_test, "f_true": f_true(x_test)})
        return train_df, test_df
    return f_true, make_1d_dataset


@app.cell
def __(mo):
    n_train = mo.ui.slider(stop=1000, value=300, start=100, label="Количество обучающих точек")
    n_test = mo.ui.slider(stop=1000, value=300, start=100, label="Количество тестовых точек")

    mo.ui.array([n_train, n_test])
    return n_test, n_train


@app.cell
def _(make_1d_dataset, n_test, n_train):
    df_train, df_test = make_1d_dataset(sigma_n=0.15, n_train=n_train.value, n_test=n_test.value)
    return df_test, df_train


@app.cell
def _(df_train):
    df_train
    return


@app.cell
def _(df_test, df_train, plt):
    fig, ax = plt.subplots()
    ax.scatter(df_train["x"], df_train["y"], label="Обучающий датасет")
    ax.plot(df_test["x"], df_test["f_true"], label="Тестовый датасет")
    ax.legend()
    return ax, fig


@app.cell
def _(np):
    def kernel(x0, x1, sigma_f=1, l=0.5):
        x0 = np.asarray(x0)[:, None]
        x1 = np.asarray(x1)[None, :]
        r = (x0 - x1)**2
        return sigma_f**2 * np.exp(-0.5 * r / l**2)
    return (kernel,)


@app.cell
def _(df_test, df_train):
    y_orig = df_train["y"].to_numpy()
    x_orig = df_train["x"].to_numpy()
    x_orig_test= df_test["x"].to_numpy()
    y_orig_test= df_test["f_true"].to_numpy()
    return x_orig, x_orig_test, y_orig, y_orig_test


@app.cell
def _(np, x_orig, x_orig_test, y_orig, y_orig_test):
    y = (y_orig - y_orig.mean()) / y_orig.std(ddof=0)
    x = (x_orig - x_orig.mean()) / x_orig.std(ddof=0)
    x_test = (x_orig_test - x_orig.mean()) / x_orig.std(ddof=0)
    y_test = (y_orig_test - y_orig.mean()) / y_orig.std(ddof=0)

    dy = np.diff(y)
    sigma_n0 = np.std(dy, ddof=1) / np.sqrt(2)

    dx = np.diff(x)
    Delta = np.median(dx)
    ell0 = 5.0 * Delta
    R = x[-1] - x[0]
    l0 = np.clip(ell0, 0.05*R, 0.5*R)

    sigma_f0 = np.sqrt(max(np.var(y, ddof=1) - sigma_n0**2, 1e-12))  # или просто 1.0
    return (
        Delta,
        R,
        dx,
        dy,
        ell0,
        l0,
        sigma_f0,
        sigma_n0,
        x,
        x_test,
        y,
        y_test,
    )


@app.cell
def _(l0, sigma_f0, sigma_n0):
    sigma_n0, sigma_f0, l0
    return


@app.cell
def _(kernel, np):
    def gp_fit(X, y, sigma_f, sigma_n, l):
        K = kernel(X, X, sigma_f=sigma_f, l=l)
        Ky = K + (sigma_n**2) * np.eye(K.shape[0])
        L = np.linalg.cholesky(Ky)
        alpha = np.linalg.solve(L, y)
        beta = np.linalg.solve(L.T, alpha)   # = Ky^{-1} y
        return L, beta

    def gp_marginal_likelihood(X, y, sigma_f, sigma_n, l):
        L, beta = gp_fit(X, y, sigma_f, sigma_n, l)
        n = y.size
        return -0.5 * (y @ beta) - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2*np.pi)
    return gp_fit, gp_marginal_likelihood


@app.cell
def _(l0, sigma_f0, sigma_n0):
    theta0 = [sigma_f0, sigma_n0, l0]
    return (theta0,)


@app.cell
def _(gp_marginal_likelihood, sc, theta0, x, y):
    res = sc.optimize.minimize(lambda theta: -gp_marginal_likelihood(
        x, y, theta[0], theta[1], theta[2]
    ), theta0, bounds=((1e-6, None), (1e-6, 10), (1e-6, 10)))
    return (res,)


@app.cell
def _(res):
    res.x
    return


@app.cell
def _(gp_marginal_likelihood, res, x, y):
    gp_marginal_likelihood(x, y, *res.x)
    return


@app.cell
def _(gp_fit, res, theta0, x, y):
    L, beta = gp_fit(x, y, *res.x)
    L0, beta0 = gp_fit(x, y, *theta0)
    return L, L0, beta, beta0


@app.cell
def _(kernel, res, theta0, x, x_test):
    K1 = kernel(x, x_test, res.x[0], res.x[2])
    K2 = kernel(x_test, x_test, res.x[0], res.x[2])
    K10 = kernel(x, x_test, theta0[0], theta0[2])
    K20 = kernel(x_test, x_test, theta0[0], theta0[2])
    return K1, K10, K2, K20


@app.cell
def _(K1, K10, beta, beta0, np):
    mu = np.dot(K1.T, beta)
    mu0 = np.dot(K10.T, beta0)
    return mu, mu0


@app.cell
def _(mu, mu0, plt, x, x_test, y, y_test):
    fig1, ax1 = plt.subplots()
    ax1.scatter(x, y, label="Обучающие данные")
    ax1.plot(x_test, mu, color="b", label="ГП на тестовых данных")
    ax1.plot(x_test, mu0, color="g", label="ГП на тестовых данных (начальное приближение)")
    ax1.scatter(x_test, y_test, label="Тестовые данные")
    return ax1, fig1


@app.cell
def _(mu, np, y_test):
    rsm = np.sum((mu - y_test)**2) / y_test.size
    return (rsm,)


@app.cell
def _(mu0, np, y_test):
    rsm0 = np.sum((mu0 - y_test)**2) / y_test.size
    return (rsm0,)


@app.cell
def _(rsm, rsm0):
    100*(rsm0 - rsm) / rsm0
    return


@app.cell
def _(K1, K2, L, np):
    V = np.linalg.solve(L, K1)
    post_cov = K2 - V.T @ V
    post_sigma = np.sqrt(np.diag(post_cov))
    return V, post_cov, post_sigma


@app.cell
def _(mu, plt, post_sigma, x, x_test, y, y_test):
    fig2, ax2 = plt.subplots()
    ax2.scatter(x, y, label="Обучающие данные")
    ax2.plot(x_test, mu, color="b", label="ГП на тестовых данных")
    ax2.fill_between(x_test, mu - 1.96 * post_sigma, mu + 1.96 * post_sigma, alpha=0.4)
    ax2.scatter(x_test, y_test, label="Тестовые данные")
    return ax2, fig2


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
