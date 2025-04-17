import marimo

__generated_with = "0.9.31"
app = marimo.App(width="medium")


@app.cell
def __():
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    import pandas as pd
    return mo, np, pd, plt


@app.cell
def __():
    # Параметры модели
    num_steps = 12 # Количество шагов
    num_samples = 50 # Количество реализаций случайного процесса
    return num_samples, num_steps


@app.cell
def __(mo):
    p_slider = mo.ui.slider(value=0.5, start=0, stop=1, step=0.01, label="Степень детерминированности: ")
    p_slider
    return (p_slider,)


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

    mean_val = np.mean(samples_positions, axis=0)
    std_val = np.std(samples_positions, axis=0)
    return dW, dt, i, j, mean_val, mu, samples_positions, std_val, steps


@app.cell
def __(mean_val, num_samples, plt, samples_positions, std_val, steps):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for s in range(num_samples):
        ax.plot(steps, samples_positions[s], label=f"Траектория {s+1}" if num_samples < 10 else None, linewidth=0.7)
    ax.plot(steps, mean_val, c='k', label=f"Среднее")
    ax.plot(steps, mean_val + 3 * std_val, c='k', linestyle='--', label=r'$M_x(t) \pm 3 \sigma(t)$')
    ax.plot(steps, mean_val - 3 * std_val, c='k', linestyle='--')
    ax.set_xlabel("Время, с")
    ax.legend()
    ax.set_ylabel(r"$X(t)$")
    return ax, fig, s


@app.cell
def __(i, pd, steps):
    df = pd.DataFrame(
        {f"t={steps[i]}"}
    )
    return (df,)


@app.cell
def __(np, samples_positions):
    cov_matrix = np.cov(samples_positions, rowvar=False)
    corr_matrix = np.corrcoef(samples_positions, rowvar=False)
    return corr_matrix, cov_matrix


@app.cell
def __(corr_matrix, np, num_steps):
    rho = np.zeros(num_steps - 1)
    for idx in range(num_steps - 1):
        rho[idx] = np.mean(np.diag(corr_matrix, idx))
    return idx, rho


@app.cell
def __(plt, rho):
    figr, axr = plt.subplots(figsize=(10, 6.5))
    axr.plot(rho)
    axr.set_xlabel("Шаг")
    axr.set_ylabel(r"$\rho(\tau)$")
    return axr, figr


@app.cell
def __(mo):
    mo.md(r"""### Траектории""")
    return


@app.cell
def __(mo, num_steps, samples_positions, steps):
    template_tnew = ''.join([f"t={steps[0]:>8.3f}"] + [f"{t:>10.3f}" for t in steps[1:]])
    template = ''.join(["{" + f"{iz}" + ":^10}" for iz in range(num_steps)])
    with mo.redirect_stdout():
        print(template_tnew)
        for iis in range(samples_positions.shape[0]):
            filler_s = [f"№{iis+1}|{samples_positions[iis, 0]:>7.3f}"] + [f"{samples_positions[iis, jjs]:>10.3f}" for jjs in range(1, samples_positions.shape[1])]
            print(template.format(*filler_s))
    return filler_s, iis, template, template_tnew


@app.cell
def __(mo):
    mo.md(r"""### Ковариационная функция $К_x(t, t')$""")
    return


@app.cell
def __(cov_matrix, mo, template, template_t):
    with mo.redirect_stdout():
        print(template_t)
        for iic in range(cov_matrix.shape[0]):
            filler_cov = ["-" if jjc < iic else f"{cov_matrix[iic, jjc]:<.3f}" for jjc in range(cov_matrix.shape[1])]
            print(template.format(*filler_cov))
    return filler_cov, iic


@app.cell
def __(mo):
    mo.md(r"""### Корелляционная функция $r_x(t, t')$""")
    return


@app.cell
def __(corr_matrix, mo, steps, template):
    template_t = ''.join([f"t={steps[0]:^6.3f}  "] + [f"{t:^10.3f}" for t in steps[1:]])
    with mo.redirect_stdout():
        print(template_t)
        for ii in range(corr_matrix.shape[0]):
            filler = ["-" if jj < ii else f"{corr_matrix[ii, jj]:<.3f}" for jj in range(corr_matrix.shape[1])]
            print(template.format(*filler))
    return filler, ii, template_t


@app.cell
def __(mo, num_steps):
    first_sample_slider = mo.ui.slider(value=1, start=1, stop=num_steps, step=1, label="Момент времени №1: ")
    second_sample_slider = mo.ui.slider(value=2, start=1, stop=num_steps, step=1, label="Момент времени №2: ")
    return first_sample_slider, second_sample_slider


@app.cell
def __(first_sample_slider, mo, second_sample_slider):
    mo.hstack(
        [first_sample_slider, second_sample_slider]
    )
    return


@app.cell
def __(first_sample_slider, plt, samples_positions, second_sample_slider):
    fig_sc, ax_sc = plt.subplots(figsize=(10, 6.5))
    x, y = samples_positions[:, first_sample_slider.value-1], samples_positions[:, second_sample_slider.value-1]
    ax_sc.scatter(x, y)
    ax_sc.set_title(
        f"Коэффициент корреляции r = {
        sum((x-x.mean()) * (y - y.mean())) / (x.size * x.std() * y.std()):.2f
        }"
    )
    return ax_sc, fig_sc, x, y


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
