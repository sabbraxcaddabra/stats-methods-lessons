import marimo

__generated_with = "0.9.31"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import scipy.stats as st
    from scipy.optimize import root_scalar
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt, root_scalar, st


@app.cell(hide_code=True)
def __(mo):
    mu_suit = mo.ui.number(
        value=5,
        start=0,
        label="Математическое ожидаение содержания масла в исправных двигателях",
    )
    mu_unsuit = mo.ui.number(
        value=7,
        start=0,
        label="Математическое ожидаение содержания масла в неисправных двигателях",
    )
    sigma_suit = mo.ui.number(
        value=1, start=0, label="СКО содержания масла в исправных двигателях"
    )
    sigma_unsuit = mo.ui.number(
        value=1.2, start=0, label="СКО содержания масла в неисправных двигателях"
    )
    P_suit = mo.ui.number(
        value=0.7, start=0, stop=1, label="Вероятность того, что двигатель исправен"
    )
    first_array = mo.ui.array(
        [
            mo.ui.number(value=0, start=0, label="$C_{11}$"),
            mo.ui.number(value=1, start=0, label="$C_{12}$"),
        ]
    )
    second_array = mo.ui.array(
        [
            mo.ui.number(value=20, start=0, label="$C_{21}$"),
            mo.ui.number(value=0, start=0, label="$C_{22}$"),
        ]
    )
    return (
        P_suit,
        first_array,
        mu_suit,
        mu_unsuit,
        second_array,
        sigma_suit,
        sigma_unsuit,
    )


@app.cell(hide_code=True)
def __(
    P_suit,
    first_array,
    mo,
    mu_suit,
    mu_unsuit,
    second_array,
    sigma_suit,
    sigma_unsuit,
):
    mo.vstack(
        [
            mu_suit,
            mu_unsuit,
            sigma_suit,
            sigma_unsuit,
            P_suit,
            mo.md("Платежная матрица"),
            mo.hstack([first_array, second_array]),
        ]
    )
    return


@app.cell(hide_code=True)
def __(
    P_suit,
    first_array,
    mu_suit,
    mu_unsuit,
    np,
    root_scalar,
    second_array,
    sigma_suit,
    sigma_unsuit,
    st,
):
    C11 = first_array.value[0]
    C12 = first_array.value[1]
    C21 = second_array.value[0]
    C22 = second_array.value[1]

    x_suit_minmax = (
        mu_suit.value - 4 * sigma_suit.value,
        mu_suit.value + 4 * sigma_suit.value,
    )
    x_unsuit_minmax = (
        mu_unsuit.value - 4 * sigma_unsuit.value,
        mu_unsuit.value + 4 * sigma_unsuit.value,
    )

    x_min = min(x_suit_minmax[0], x_unsuit_minmax[0])
    x_max = max(x_suit_minmax[1], x_unsuit_minmax[1])

    x = np.linspace(x_min, x_max, 1000)

    y_suit = st.norm.pdf(x, loc=mu_suit.value, scale=sigma_suit.value)
    y_unsuit = st.norm.pdf(x, loc=mu_unsuit.value, scale=sigma_unsuit.value)

    Rf = (
        lambda X0, P_suit: C11
        * P_suit
        * st.norm.cdf(X0, loc=mu_suit.value, scale=sigma_suit.value)
        + C12 * P_suit * st.norm.sf(X0, loc=mu_suit.value, scale=sigma_suit.value)
        + C21
        * (1 - P_suit)
        * st.norm.cdf(X0, loc=mu_unsuit.value, scale=sigma_unsuit.value)
        + C22
        * (1 - P_suit)
        * st.norm.sf(X0, loc=mu_unsuit.value, scale=sigma_unsuit.value)
    )

    RfX0 = (
        lambda X0, P_suit: C11
        * P_suit
        * st.norm.pdf(X0, loc=mu_suit.value, scale=sigma_suit.value)
        - C12 * P_suit * st.norm.pdf(X0, loc=mu_suit.value, scale=sigma_suit.value)
        + C21
        * (1 - P_suit)
        * st.norm.pdf(X0, loc=mu_unsuit.value, scale=sigma_unsuit.value)
        - C22
        * (1 - P_suit)
        * st.norm.pdf(X0, loc=mu_unsuit.value, scale=sigma_unsuit.value)
    )

    R_P_suit_deriv = (
        lambda X0: C11 * st.norm.cdf(X0, loc=mu_suit.value, scale=sigma_suit.value)
        + C12 * st.norm.sf(X0, loc=mu_suit.value, scale=sigma_suit.value)
        - C21 * st.norm.cdf(X0, loc=mu_unsuit.value, scale=sigma_unsuit.value)
        - C22 * st.norm.sf(X0, loc=mu_unsuit.value, scale=sigma_unsuit.value)
    )

    R = Rf(x, P_suit.value)
    R_deriv = RfX0(x, P_suit.value)

    X0_baes = root_scalar(
        lambda X0: RfX0(X0, P_suit.value), x0=0.5 * (mu_suit.value + mu_unsuit.value)
    ).root
    X0_minimax = root_scalar(
        lambda X0: R_P_suit_deriv(X0), x0=0.5 * (mu_suit.value + mu_unsuit.value)
    ).root
    return (
        C11,
        C12,
        C21,
        C22,
        R,
        R_P_suit_deriv,
        R_deriv,
        Rf,
        RfX0,
        X0_baes,
        X0_minimax,
        x,
        x_max,
        x_min,
        x_suit_minmax,
        x_unsuit_minmax,
        y_suit,
        y_unsuit,
    )


@app.cell(hide_code=True)
def __(X0_baes, X0_minimax, mo):
    mo.md(f"""
    Решение методом Байеса - $X_0 = {X0_baes:.2f}$

    Решение методом минимакс - $X_0 = {X0_minimax:.2f}$
    """)
    return


@app.cell(hide_code=True)
def __(X0_baes, X0_minimax, plt, x, y_suit, y_unsuit):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(x, y_suit, label="Распределение для исправных двигателей")
    ax.plot(x, y_unsuit, label="Распределение для неисправных двигателей")
    ax.vlines(
        X0_baes,
        0,
        1,
        label="Решающее правило методом Байеса",
        linestyle="dashed",
        color="k",
    )
    ax.vlines(
        X0_minimax,
        0,
        1,
        label="Решающее правило методом минимакс",
        linestyle="dashed",
        color="r",
    )
    ax.set_xlabel("Содержание железоуглерода в масле")
    ax.set_ylabel("Плотность распределения")
    ax.set_ylim(0, 1.05 * max(y_suit.max(), y_unsuit.max()))
    ax.legend()
    return ax, fig


@app.cell(hide_code=True)
def __(R, plt, x):
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    ax2.plot(x, R)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(15))
    ax2.set_xlabel("Содержание железоуглерода в масле (решающее правило)")
    ax2.set_ylabel("Рискк")
    ax2.grid()
    plt.show()
    return ax2, fig2


@app.cell(hide_code=True)
def __(R_deriv, plt, x):
    fig3, ax3 = plt.subplots(figsize=(14, 7))
    ax3.plot(x, R_deriv)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(20))
    ax3.set_xlabel("Содержание железоуглерода в масле (решающее правило)")
    ax3.set_ylabel("Производная риска")
    ax3.grid()
    plt.show()
    return ax3, fig3


@app.cell
def __(mo):
    x0_slider = mo.ui.slider(start=1, stop=10, step=0.01)
    x0_slider
    return (x0_slider,)


@app.cell(hide_code=True)
def __(Rf, np, plt, second_array, x0_slider):
    p_suit = np.linspace(0, 1)
    fig1, ax1 = plt.subplots()
    ax1.set_ylim(0, second_array.value[0])
    ax1.plot(p_suit, Rf(x0_slider.value, p_suit))
    return ax1, fig1, p_suit


if __name__ == "__main__":
    app.run()
