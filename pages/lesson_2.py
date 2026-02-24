import marimo

__generated_with = "0.9.31"
app = marimo.App()


@app.cell
def __():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from scipy.integrate import solve_ivp
    from dataclasses import dataclass
    import pandas as pd
    from scipy.optimize import newton
    import scipy.stats as st
    import marimo as mo
    import matplotlib.ticker as tck
    import sys
    from pathlib import Path

    for _candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (_candidate / "lesson2_rhs.py").exists():
            if str(_candidate) not in sys.path:
                sys.path.insert(0, str(_candidate))
            break
    from lesson2_rhs import rhs as penetration_rhs, x_end as penetration_x_end

    return (
        cm,
        dataclass,
        mo,
        newton,
        np,
        pd,
        penetration_rhs,
        penetration_x_end,
        plt,
        solve_ivp,
        st,
        tck,
    )


@app.cell(hide_code=True)
def __(np):
    def get_u0(a_u, l_u, a_p, l_p, rho_u, rho_p, vc):
        rho = rho_p / rho_u
        a0 = l_u - l_p * rho
        b0 = 2 * l_u * vc + a_u + a_p * rho
        c0 = a_u * vc + l_u * vc * vc

        u0 = (b0 - np.sqrt(b0 * b0 - 4 * a0 * c0)) / (2 * a0)
        return u0

    return (get_u0,)


@app.cell(hide_code=True)
def __(dataclass, penetration_rhs):
    @dataclass
    class Penetration:
        sigmat_u: float
        c_zv_u: float
        Cu: float
        Cp: float
        A: float
        Dp: float
        Rg: float
        gamma: float
        rho_u: float

        def rs(self, t, y):
            u, v, L, x = y
            return penetration_rhs(
                u,
                v,
                L,
                self.sigmat_u,
                self.c_zv_u,
                self.Cu,
                self.Cp,
                self.A,
                self.Dp,
                self.Rg,
                self.gamma,
                self.rho_u,
            )

    return (Penetration,)


@app.cell(hide_code=True)
def __(Penetration, get_u0, np, solve_ivp):
    def model(data: dict):
        da = data["da"]
        la = data["la"]
        vc = data["vc"]
        E_p = data["E_p"]
        mu_p = data["mu_p"]
        lam_p = data["lam_p"]

        E_u = data["E_u"]

        sigmat_u = data["sigmat_u"]
        sigmat_p = data["sigmat_p"]

        rho_u = data["rho_u"]
        rho_p = data["rho_p"]

        c_zv_u = np.sqrt(E_u / rho_u)
        a_u, l_u = data["a_u"], data["l_u"]
        a_p, l_p = data["a_p"], data["l_p"]

        n_points = data.get("n_points", 500)

        gamma = np.cbrt(E_p / (3 * (1 - mu_p) * sigmat_p))
        A = 2 * (
            sigmat_p * np.log(gamma)
            + (2 / 3)
            * E_p
            * (
                (1 / 18)
                * np.pi
                * np.pi
                * (1 - lam_p)
                * (1 - 1 / np.cbrt(gamma)) ** (0.62)
                + 0.5 * sigmat_p / (E_p * lam_p)
            )
        )

        u0 = get_u0(a_u, l_u, a_p, l_p, rho_u, rho_p, vc)

        Rg = 0.5 * da * (1 + np.sqrt((2 * rho_u * (vc - u0) ** 2) / A))

        Dp = rho_p * Rg * (gamma - 1) / (gamma + 1)

        Cu = 0.5 * rho_u
        Cp = 0.5 * rho_p

        pen = Penetration(sigmat_u, c_zv_u, Cu, Cp, A, Dp, Rg, gamma, rho_u)

        y0 = np.array([u0, vc, la, 0])
        zero_u = lambda t, y: y[0]
        zero_v = lambda t, y: y[1]
        zero_L = lambda t, y: y[2]

        zero_u.terminal = True
        zero_v.terminal = True
        zero_L.terminal = True

        zero_u.direction = -1
        zero_v.direction = -1
        zero_L.direction = -1

        sol = solve_ivp(
            pen.rs,
            (0, 1),
            y0,
            events=(zero_u, zero_v, zero_L),
        )

        t = sol.t
        sol = sol.y
        return dict(t=t, u=sol[0], v=sol[1], L=sol[2], x=sol[3])

    return (model,)


@app.cell
def __(mo):
    da_label = mo.ui.number(label="Диаметр активной части, мм", value=12)
    la_label = mo.ui.number(label="Длина активной части, мм", value=160)
    rho_u_label = mo.ui.number(label="Плотность ударника, г/см^3", value=17)
    sigmat_u_label = mo.ui.number(label="Предел текучести ударника, МПа", value=1400)
    h_label = mo.ui.number(label="Толщина преграды, мм", value=140)

    da_label_p = mo.ui.number(label="$\\pm$", value=5)
    la_label_p = mo.ui.number(label="$\\pm$", value=5)
    rho_u_label_p = mo.ui.number(label="$\\pm$", value=5)
    sigmat_u_label_p = mo.ui.number(label="$\\pm$", value=5)
    h_label_p = mo.ui.number(label="$\\pm$", value=5)

    n_samples = mo.ui.number(
        label="Количество экспериментов", value=500, start=1, step=50, stop=1000
    )
    n_bins = mo.ui.number(
        label="Количество диапазонов", value=13, start=5, step=1, stop=25
    )
    return (
        da_label,
        da_label_p,
        h_label,
        h_label_p,
        la_label,
        la_label_p,
        n_bins,
        n_samples,
        rho_u_label,
        rho_u_label_p,
        sigmat_u_label,
        sigmat_u_label_p,
    )


@app.cell
def __(
    da_label,
    da_label_p,
    h_label,
    h_label_p,
    la_label,
    la_label_p,
    mo,
    n_bins,
    n_samples,
    rho_u_label,
    rho_u_label_p,
    sigmat_u_label,
    sigmat_u_label_p,
):
    mo.vstack(
        [
            mo.hstack([da_label, da_label_p]),
            mo.hstack([la_label, la_label_p]),
            mo.hstack([rho_u_label, rho_u_label_p]),
            mo.hstack([sigmat_u_label, sigmat_u_label_p]),
            mo.hstack([h_label, h_label_p]),
            n_samples,
            n_bins,
        ]
    )
    return


@app.cell(hide_code=True)
def __():
    E_p = 207e9  # Модуль упругости преграды, Па
    mu_p = 0.33  # Коэффициент Пуассона преграды, -
    lam_p = 0.997  # Коэффициент разупрочнения преграды, -
    E_u = 350e9  # Модуль упругости ударника, Па
    sigmat_p = 970e6  # предел текучести преграды, Па
    rho_p = 7850  # Плотность преграды, кг/м3
    a_u, l_u = 3.83e3, 1.5  # Коэффициенты ЛУА ударника, м/с, -
    a_p, l_p = 4.5e3, 1.49  # Коэффициенты ЛУА преграды, м/с, -
    return E_p, E_u, a_p, a_u, l_p, l_u, lam_p, mu_p, rho_p, sigmat_p


@app.cell
def __(
    E_p,
    E_u,
    a_p,
    a_u,
    da_label,
    l_p,
    l_u,
    la_label,
    lam_p,
    mu_p,
    rho_p,
    rho_u_label,
    sigmat_p,
    sigmat_u_label,
):
    base_data = dict(
        da=da_label.value * 1e-3,
        la=la_label.value * 1e-3,
        vc=1300,
        E_p=E_p,
        E_u=E_u,
        mu_p=mu_p,
        lam_p=lam_p,
        sigmat_u=sigmat_u_label.value * 1e6,
        sigmat_p=sigmat_p,
        rho_u=rho_u_label.value * 1e3,
        rho_p=rho_p,
        a_u=a_u,
        l_u=l_u,
        a_p=a_p,
        l_p=l_p,
    )
    return (base_data,)


@app.cell
def __(
    base_data,
    da_label_p,
    h_label,
    h_label_p,
    la_label_p,
    n_samples,
    np,
    rho_u_label_p,
    sigmat_u_label_p,
):
    da_data = np.random.uniform(
        base_data["da"] - 1e-2 * da_label_p.value * base_data["da"],
        base_data["da"] + 1e-2 * da_label_p.value * base_data["da"],
        n_samples.value,
    )

    la_data = np.random.uniform(
        base_data["la"] - 1e-2 * la_label_p.value * base_data["la"],
        base_data["la"] + 1e-2 * la_label_p.value * base_data["la"],
        n_samples.value,
    )

    sigmat_u_data = np.random.uniform(
        base_data["sigmat_u"] - 1e-2 * sigmat_u_label_p.value * base_data["sigmat_u"],
        base_data["sigmat_u"] + 1e-2 * sigmat_u_label_p.value * base_data["sigmat_u"],
        n_samples.value,
    )

    rho_u_data = np.random.uniform(
        base_data["rho_u"] - 1e-2 * rho_u_label_p.value * base_data["rho_u"],
        base_data["rho_u"] + 1e-2 * rho_u_label_p.value * base_data["rho_u"],
        n_samples.value,
    )

    h_data = np.random.uniform(
        h_label.value - 1e-2 * h_label_p.value * h_label.value,
        h_label.value + 1e-2 * h_label_p.value * h_label.value,
        n_samples.value,
    )
    return da_data, h_data, la_data, rho_u_data, sigmat_u_data


@app.cell
def __():
    def update_and_return_dict(base_data: dict, vc: float):
        data = base_data.copy()
        data["vc"] = vc
        return data

    return (update_and_return_dict,)


@app.cell
def __(
    base_data,
    da_data,
    h_data,
    la_data,
    n_samples,
    newton,
    np,
    penetration_x_end,
    rho_u_data,
    sigmat_u_data,
):
    ballistic_impact = np.zeros_like(da_data)
    for i in range(n_samples.value):
        da = da_data[i]
        la = la_data[i]
        sigmat_u = sigmat_u_data[i]
        rho_u = rho_u_data[i]
        bi = newton(
            lambda vc: (
                penetration_x_end(
                    vc,
                    da,
                    la,
                    base_data["E_p"],
                    base_data["mu_p"],
                    base_data["lam_p"],
                    base_data["E_u"],
                    sigmat_u,
                    base_data["sigmat_p"],
                    rho_u,
                    base_data["rho_p"],
                    base_data["a_u"],
                    base_data["l_u"],
                    base_data["a_p"],
                    base_data["l_p"],
                )
                - 1e-3 * h_data[i]
            ),
            x0=1300,
            rtol=1e-2,
        )
        ballistic_impact[i] = bi
    return (ballistic_impact,)


@app.cell
def __(ballistic_impact, n_bins, plt):
    fig, ax = plt.subplots()
    ax.hist(ballistic_impact, bins=n_bins.value, rwidth=0.9)
    ax.set_ylabel("Количество")
    ax.set_xlabel(r"$V_{ПСП}$")
    ax
    return ax, fig


@app.cell
def __(ballistic_impact, n_bins, n_samples, np, st):
    mu, sigma = np.mean(ballistic_impact), np.std(ballistic_impact)
    observed, bins = np.histogram(ballistic_impact, n_bins.value)
    expected = np.diff(st.norm.cdf(bins, loc=mu, scale=sigma)) * n_samples.value
    return bins, expected, mu, observed, sigma


@app.cell
def __(expected, observed, st):
    chi2_res = st.chisquare(observed, expected, ddof=2, sum_check=False)
    return (chi2_res,)


@app.cell(hide_code=True)
def __(ballistic_impact, mo, n_bins, np):
    count, bins_values = np.histogram(ballistic_impact, n_bins.value)
    mo.ui.table(
        data=[
            {
                "Диапазон скоростей, м/с": f"{bins_values[i]:.1f} - {bins_values[i + 1]:.1f}",
                "Количество": count[i],
            }
            for i in range(len(count))
        ],
        label="Результаты",
        page_size=30,
    )
    return bins_values, count


@app.cell
def __(chi2_res, n_bins, newton, np, plt, st, tck):
    chi2_max = np.ceil(
        newton(
            lambda chi2: 1e-4 - st.chi2.sf(chi2, df=n_bins.value - 1 - 2),
            x0=chi2_res.statistic,
        )
    )
    fig_d, ax_d = plt.subplots()
    ax_d.plot(
        np.linspace(0, chi2_max),
        st.chi2.cdf(np.linspace(0, chi2_max), df=n_bins.value - 1 - 2),
    )
    ax_d.set_xlim(0, chi2_max)
    ax_d.set_ylim(0, 1.05)
    ax_d.set_xticks(np.arange(0, chi2_max, 2))
    ax_d.set_xlabel(r"$\chi^2$")
    ax_d.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax_d.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax_d.grid()
    ax_d.set_ylabel(r"$F(\chi^2)$")
    return ax_d, chi2_max, fig_d


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
