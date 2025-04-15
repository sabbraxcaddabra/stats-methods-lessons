import marimo

__generated_with = "0.9.31"
app = marimo.App(width="medium")


@app.cell
def __():
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    return mo, np, plt


@app.cell
def __():
    # Параметры модели
    num_steps = 12# Количество шагов
    num_samples = 5 # Количество реализаций случайного процесса
    return num_samples, num_steps


@app.cell
def __(mo):
    p_slider = mo.ui.slider(value=0.5, start=0, stop=1, step=0.01)
    p_slider
    return (p_slider,)


@app.cell
def __(p_slider):
    p_slider.value
    return


@app.cell
def __(np, num_samples, num_steps, p_slider):
    mu = 1
    dt = 0.1
    # Массив для хранения позиций частицы на каждом шаге
    samples_positions = np.zeros((num_samples, num_steps))
    samples_positions[:, 0] = np.random.normal(0, 0.1, size=num_samples)
    steps = np.linspace(0, num_steps * dt, num_steps)
    for i in range(num_samples):
        # Моделирование случайного блуждания
        for j in range(1, num_steps):
            # Случайный шаг: -1 (влево) или 1 (вправо)
            dW = np.random.normal(0, np.sqrt(dt))
            # Обновление позиции
            samples_positions[i, j] = samples_positions[i, j - 1] + p_slider.value * mu * dt + (1 - p_slider.value) * dW
    return dW, dt, i, j, mu, samples_positions, steps


@app.cell
def __(num_samples, plt, samples_positions, steps):
    fig, ax = plt.subplots()
    for s in range(num_samples):
        ax.plot(steps, samples_positions[s], label=f"Траектория {s+1}")
    ax.set_xlabel("Время, с")
    ax.legend()
    ax.set_ylabel(r"$X(t)$")
    return ax, fig, s


@app.cell
def __(np, samples_positions):
    cov_matrix = np.cov(samples_positions, rowvar=False)
    return (cov_matrix,)


@app.cell
def __(np, samples_positions):
    corr_matrix = np.corrcoef(samples_positions, rowvar=False)
    return (corr_matrix,)


@app.cell
def __(corr_matrix, np, num_steps):
    rho = np.zeros(num_steps - 1)
    for idx in range(num_steps - 1):
        rho[idx] = np.mean(np.diag(corr_matrix, idx))
    return idx, rho


@app.cell
def __(plt, rho):
    figr, axr = plt.subplots()
    axr.plot(rho)
    axr.set_xlabel("Шаг")
    axr.set_ylabel(r"$\rho(\tau)$")
    return axr, figr


@app.cell
def __(corr_matrix):
    corr_matrix
    return


@app.cell
def __(np, steps):
    np.cumsum(np.diff(steps))
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
