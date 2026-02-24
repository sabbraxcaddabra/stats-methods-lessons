#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

static inline double safe_denom(double value) {
    const double eps = 1e-12;
    if (fabs(value) < eps) {
        return value < 0.0 ? -eps : eps;
    }
    return value;
}

static inline void compute_rhs(
    double u,
    double v,
    double l_value,
    double sigmat_u,
    double c_zv_u,
    double cu,
    double cp,
    double a_value,
    double dp,
    double rg,
    double gamma,
    double rho_u,
    double *u_dot,
    double *v_dot,
    double *l_dot,
    double *x_dot
) {
    const double u_safe = safe_denom(u);
    const double s_value = 0.5 * rg * (v / u_safe - 1.0) * (1.0 - 1.0 / (gamma * gamma));
    const double du = rho_u * s_value;

    const double u_num = sigmat_u * (1.0 + (v - u) / c_zv_u)
        + cu * (v - u) * (v - u)
        - a_value
        - cp * u * u;
    double local_u_dot = u_num / safe_denom(dp + du);
    if (local_u_dot > 0.0) {
        local_u_dot = 0.0;
    }

    double local_v_dot =
        -sigmat_u * (1.0 + (v - u) / c_zv_u) / safe_denom(rho_u * (l_value - s_value));
    if (local_v_dot > 0.0) {
        local_v_dot = 0.0;
    }

    *u_dot = local_u_dot;
    *v_dot = local_v_dot;
    *l_dot = u - v;
    *x_dot = u;
}

static inline double estimate_step(
    double u,
    double v,
    double l_value,
    double u_dot,
    double v_dot,
    double l_dot,
    double max_step
) {
    const double min_step = fmax(1e-9, max_step * 1e-4);
    double tau = 1e12;
    int has_tau = 0;
    if (fabs(u_dot) > 1e-16) {
        tau = fmin(tau, fabs(u / u_dot));
        has_tau = 1;
    }
    if (fabs(v_dot) > 1e-16) {
        tau = fmin(tau, fabs(v / v_dot));
        has_tau = 1;
    }
    if (fabs(l_dot) > 1e-16) {
        tau = fmin(tau, fabs(l_value / l_dot));
        has_tau = 1;
    }
    double h = has_tau ? (0.05 * tau) : max_step;
    if (!isfinite(h) || h <= 0.0) {
        h = min_step;
    }
    if (h < min_step) {
        h = min_step;
    }
    if (h > max_step) {
        h = max_step;
    }
    return h;
}

static inline double get_u0(
    double a_u,
    double l_u,
    double a_p,
    double l_p,
    double rho_u,
    double rho_p,
    double vc
) {
    const double rho_ratio = rho_p / rho_u;
    const double a0 = l_u - l_p * rho_ratio;
    const double b0 = 2.0 * l_u * vc + a_u + a_p * rho_ratio;
    const double c0 = a_u * vc + l_u * vc * vc;
    if (fabs(a0) < 1e-12) {
        return -c0 / safe_denom(b0);
    }
    double disc = b0 * b0 - 4.0 * a0 * c0;
    if (disc < 0.0) {
        disc = 0.0;
    }
    return (b0 - sqrt(disc)) / (2.0 * a0);
}

static PyObject *rhs(PyObject *self, PyObject *args) {
    double u = 0.0;
    double v = 0.0;
    double l_value = 0.0;
    double sigmat_u = 0.0;
    double c_zv_u = 0.0;
    double cu = 0.0;
    double cp = 0.0;
    double a_value = 0.0;
    double dp = 0.0;
    double rg = 0.0;
    double gamma = 0.0;
    double rho_u = 0.0;

    if (!PyArg_ParseTuple(
            args,
            "dddddddddddd",
            &u,
            &v,
            &l_value,
            &sigmat_u,
            &c_zv_u,
            &cu,
            &cp,
            &a_value,
            &dp,
            &rg,
            &gamma,
            &rho_u
        )) {
        return NULL;
    }

    double u_dot = 0.0;
    double v_dot = 0.0;
    double l_dot = 0.0;
    double x_dot = 0.0;
    compute_rhs(
        u,
        v,
        l_value,
        sigmat_u,
        c_zv_u,
        cu,
        cp,
        a_value,
        dp,
        rg,
        gamma,
        rho_u,
        &u_dot,
        &v_dot,
        &l_dot,
        &x_dot
    );
    return Py_BuildValue("(dddd)", u_dot, v_dot, l_dot, x_dot);
}

static PyObject *x_end(PyObject *self, PyObject *args) {
    double vc = 0.0;
    double da = 0.0;
    double la = 0.0;
    double E_p = 0.0;
    double mu_p = 0.0;
    double lam_p = 0.0;
    double E_u = 0.0;
    double sigmat_u = 0.0;
    double sigmat_p = 0.0;
    double rho_u = 0.0;
    double rho_p = 0.0;
    double a_u = 0.0;
    double l_u = 0.0;
    double a_p = 0.0;
    double l_p = 0.0;
    double max_step = 1e-4;
    double max_time = 1.0;
    int max_steps = 200000;

    if (!PyArg_ParseTuple(
            args,
            "ddddddddddddddd|ddi",
            &vc,
            &da,
            &la,
            &E_p,
            &mu_p,
            &lam_p,
            &E_u,
            &sigmat_u,
            &sigmat_p,
            &rho_u,
            &rho_p,
            &a_u,
            &l_u,
            &a_p,
            &l_p,
            &max_step,
            &max_time,
            &max_steps
        )) {
        return NULL;
    }

    const double c_zv_u = sqrt(E_u / rho_u);
    const double gamma = cbrt(E_p / (3.0 * (1.0 - mu_p) * sigmat_p));
    const double gamma_cbrt = cbrt(gamma);
    const double A = 2.0 * (
        sigmat_p * log(gamma) + (2.0 / 3.0) * E_p * (
            (1.0 / 18.0) * M_PI * M_PI * (1.0 - lam_p) * pow(1.0 - 1.0 / gamma_cbrt, 0.62)
            + 0.5 * sigmat_p / (E_p * lam_p)
        )
    );
    const double u0 = get_u0(a_u, l_u, a_p, l_p, rho_u, rho_p, vc);
    const double rg = 0.5 * da * (1.0 + sqrt((2.0 * rho_u * (vc - u0) * (vc - u0)) / A));
    const double dp = rho_p * rg * (gamma - 1.0) / (gamma + 1.0);
    const double cu = 0.5 * rho_u;
    const double cp = 0.5 * rho_p;

    double u = u0;
    double v = vc;
    double l_value = la;
    double x = 0.0;
    double t = 0.0;

    for (int step = 0; step < max_steps; ++step) {
        if (t >= max_time || u <= 0.0 || v <= 0.0 || l_value <= 0.0) {
            break;
        }

        double k1u = 0.0;
        double k1v = 0.0;
        double k1l = 0.0;
        double k1x = 0.0;
        compute_rhs(
            u, v, l_value, sigmat_u, c_zv_u, cu, cp, A, dp, rg, gamma, rho_u,
            &k1u, &k1v, &k1l, &k1x
        );

        double h = estimate_step(u, v, l_value, k1u, k1v, k1l, max_step);
        if (t + h > max_time) {
            h = max_time - t;
        }
        if (h <= 0.0) {
            break;
        }

        double k2u = 0.0;
        double k2v = 0.0;
        double k2l = 0.0;
        double k2x = 0.0;
        compute_rhs(
            u + 0.5 * h * k1u,
            v + 0.5 * h * k1v,
            l_value + 0.5 * h * k1l,
            sigmat_u,
            c_zv_u,
            cu,
            cp,
            A,
            dp,
            rg,
            gamma,
            rho_u,
            &k2u,
            &k2v,
            &k2l,
            &k2x
        );

        double k3u = 0.0;
        double k3v = 0.0;
        double k3l = 0.0;
        double k3x = 0.0;
        compute_rhs(
            u + 0.5 * h * k2u,
            v + 0.5 * h * k2v,
            l_value + 0.5 * h * k2l,
            sigmat_u,
            c_zv_u,
            cu,
            cp,
            A,
            dp,
            rg,
            gamma,
            rho_u,
            &k3u,
            &k3v,
            &k3l,
            &k3x
        );

        double k4u = 0.0;
        double k4v = 0.0;
        double k4l = 0.0;
        double k4x = 0.0;
        compute_rhs(
            u + h * k3u,
            v + h * k3v,
            l_value + h * k3l,
            sigmat_u,
            c_zv_u,
            cu,
            cp,
            A,
            dp,
            rg,
            gamma,
            rho_u,
            &k4u,
            &k4v,
            &k4l,
            &k4x
        );

        u += h * (k1u + 2.0 * k2u + 2.0 * k3u + k4u) / 6.0;
        v += h * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0;
        l_value += h * (k1l + 2.0 * k2l + 2.0 * k3l + k4l) / 6.0;
        x += h * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0;

        if (!isfinite(u) || !isfinite(v) || !isfinite(l_value) || !isfinite(x)) {
            break;
        }
        t += h;
    }

    return PyFloat_FromDouble(x);
}

static PyMethodDef Lesson2Methods[] = {
    {"rhs", rhs, METH_VARARGS, "Compute RHS for lesson 2 penetration ODE."},
    {"x_end", x_end, METH_VARARGS, "Integrate penetration ODE and return final x."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef lesson2_module = {
    PyModuleDef_HEAD_INIT,
    "lesson2_native",
    "Native acceleration module for lesson 2 penetration model.",
    -1,
    Lesson2Methods
};

PyMODINIT_FUNC PyInit_lesson2_native(void) {
    return PyModule_Create(&lesson2_module);
}
