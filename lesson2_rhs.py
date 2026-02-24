from __future__ import annotations

import importlib.util
import math
import shutil
import subprocess
import sysconfig
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Callable


NativeRhsFn = Callable[
    [float, float, float, float, float, float, float, float, float, float, float, float],
    tuple[float, float, float, float],
]
NativeXEndFn = Callable[
    [
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        int,
    ],
    float,
]


_ROOT_DIR = Path(__file__).resolve().parent
_NATIVE_SOURCE = _ROOT_DIR / "native" / "lesson2_native.c"
_EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
_NATIVE_LIBRARY = Path(tempfile.gettempdir()) / f"lesson2_native{_EXT_SUFFIX}"


def _needs_rebuild() -> bool:
    if not _NATIVE_SOURCE.exists():
        return False
    if not _NATIVE_LIBRARY.exists():
        return True
    return _NATIVE_LIBRARY.stat().st_mtime < _NATIVE_SOURCE.stat().st_mtime


def _build_native_extension() -> None:
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None or not _NATIVE_SOURCE.exists():
        return

    include_paths = {
        sysconfig.get_config_var("INCLUDEPY"),
        sysconfig.get_paths().get("include"),
        sysconfig.get_paths().get("platinclude"),
    }
    include_args: list[str] = []
    for include_path in sorted(path for path in include_paths if path):
        include_args.extend(["-I", include_path])

    _NATIVE_LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            compiler,
            "-Ofast",
            "-march=native",
            "-flto",
            "-shared",
            "-fPIC",
            str(_NATIVE_SOURCE),
            *include_args,
            "-o",
            str(_NATIVE_LIBRARY),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _load_dynamic_native_module() -> ModuleType | None:
    if _needs_rebuild():
        try:
            _build_native_extension()
        except Exception:
            return None
    if not _NATIVE_LIBRARY.exists():
        return None

    module_name = "lesson2_native"
    spec = importlib.util.spec_from_file_location(module_name, _NATIVE_LIBRARY)
    if spec is None or spec.loader is None:
        return None
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        try:
            _NATIVE_LIBRARY.unlink(missing_ok=True)
        except Exception:
            pass
        return None
    return module


def _load_native_module() -> ModuleType | None:
    try:
        import lesson2_native as imported_module
    except Exception:
        imported_module = None

    if (
        imported_module is not None
        and callable(getattr(imported_module, "rhs", None))
        and callable(getattr(imported_module, "x_end", None))
    ):
        return imported_module
    return _load_dynamic_native_module()


_native_module = _load_native_module()
_native_rhs: NativeRhsFn | None = None
_native_x_end: NativeXEndFn | None = None
if _native_module is not None:
    candidate_rhs = getattr(_native_module, "rhs", None)
    candidate_x_end = getattr(_native_module, "x_end", None)
    if callable(candidate_rhs):
        _native_rhs = candidate_rhs
    if callable(candidate_x_end):
        _native_x_end = candidate_x_end


def _safe_denom(value: float, eps: float = 1e-12) -> float:
    if abs(value) < eps:
        return -eps if value < 0.0 else eps
    return value


def _get_u0(
    a_u: float,
    l_u: float,
    a_p: float,
    l_p: float,
    rho_u: float,
    rho_p: float,
    vc: float,
) -> float:
    rho_ratio = rho_p / rho_u
    a0 = l_u - l_p * rho_ratio
    b0 = 2.0 * l_u * vc + a_u + a_p * rho_ratio
    c0 = a_u * vc + l_u * vc * vc
    if abs(a0) < 1e-12:
        return -c0 / _safe_denom(b0)
    discriminant = b0 * b0 - 4.0 * a0 * c0
    if discriminant < 0.0:
        discriminant = 0.0
    return (b0 - math.sqrt(discriminant)) / (2.0 * a0)


def rhs_python(
    u: float,
    v: float,
    l_value: float,
    sigmat_u: float,
    c_zv_u: float,
    cu: float,
    cp: float,
    a_value: float,
    dp: float,
    rg: float,
    gamma: float,
    rho_u: float,
) -> tuple[float, float, float, float]:
    u_safe = _safe_denom(u)
    s_value = 0.5 * rg * (v / u_safe - 1.0) * (1.0 - 1.0 / gamma**2)
    du = rho_u * s_value

    u_num = (
        sigmat_u * (1.0 + (v - u) / c_zv_u)
        + cu * (v - u) ** 2
        - a_value
        - cp * u**2
    )
    u_dot = u_num / _safe_denom(dp + du)
    if u_dot > 0.0:
        u_dot = 0.0

    v_dot = -sigmat_u * (1.0 + (v - u) / c_zv_u) / _safe_denom(rho_u * (l_value - s_value))
    if v_dot > 0.0:
        v_dot = 0.0

    return (u_dot, v_dot, u - v, u)


def rhs(
    u: float,
    v: float,
    l_value: float,
    sigmat_u: float,
    c_zv_u: float,
    cu: float,
    cp: float,
    a_value: float,
    dp: float,
    rg: float,
    gamma: float,
    rho_u: float,
) -> tuple[float, float, float, float]:
    native_rhs: NativeRhsFn | None = _native_rhs
    if native_rhs is not None:
        return native_rhs(
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
        )
    return rhs_python(
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
    )


def _estimate_step(
    u: float,
    v: float,
    l_value: float,
    u_dot: float,
    v_dot: float,
    l_dot: float,
    max_step: float,
) -> float:
    min_step = max(1e-9, max_step * 1e-4)
    tau_candidates = []
    if abs(u_dot) > 1e-16:
        tau_candidates.append(abs(u / u_dot))
    if abs(v_dot) > 1e-16:
        tau_candidates.append(abs(v / v_dot))
    if abs(l_dot) > 1e-16:
        tau_candidates.append(abs(l_value / l_dot))
    if tau_candidates:
        step = 0.05 * min(tau_candidates)
    else:
        step = max_step
    if not math.isfinite(step) or step <= 0.0:
        step = min_step
    return min(max(step, min_step), max_step)


def x_end_python(
    vc: float,
    da: float,
    la: float,
    E_p: float,
    mu_p: float,
    lam_p: float,
    E_u: float,
    sigmat_u: float,
    sigmat_p: float,
    rho_u: float,
    rho_p: float,
    a_u: float,
    l_u: float,
    a_p: float,
    l_p: float,
    max_step: float = 1e-4,
    max_time: float = 1.0,
    max_steps: int = 200_000,
) -> float:
    c_zv_u = math.sqrt(E_u / rho_u)
    gamma = (E_p / (3.0 * (1.0 - mu_p) * sigmat_p)) ** (1.0 / 3.0)
    gamma_cbrt = gamma ** (1.0 / 3.0)
    a_value = 2.0 * (
        sigmat_p * math.log(gamma)
        + (2.0 / 3.0)
        * E_p
        * (
            (1.0 / 18.0)
            * math.pi
            * math.pi
            * (1.0 - lam_p)
            * (1.0 - 1.0 / gamma_cbrt) ** 0.62
            + 0.5 * sigmat_p / (E_p * lam_p)
        )
    )

    u0 = _get_u0(a_u, l_u, a_p, l_p, rho_u, rho_p, vc)
    rg = 0.5 * da * (1.0 + math.sqrt((2.0 * rho_u * (vc - u0) ** 2) / a_value))
    dp = rho_p * rg * (gamma - 1.0) / (gamma + 1.0)
    cu = 0.5 * rho_u
    cp = 0.5 * rho_p

    u = u0
    v = vc
    l_value = la
    x = 0.0
    t = 0.0
    for _ in range(max_steps):
        if t >= max_time or u <= 0.0 or v <= 0.0 or l_value <= 0.0:
            break

        k1u, k1v, k1l, k1x = rhs_python(
            u, v, l_value, sigmat_u, c_zv_u, cu, cp, a_value, dp, rg, gamma, rho_u
        )
        h = _estimate_step(u, v, l_value, k1u, k1v, k1l, max_step)
        if t + h > max_time:
            h = max_time - t
        if h <= 0.0:
            break

        k2u, k2v, k2l, k2x = rhs_python(
            u + 0.5 * h * k1u,
            v + 0.5 * h * k1v,
            l_value + 0.5 * h * k1l,
            sigmat_u,
            c_zv_u,
            cu,
            cp,
            a_value,
            dp,
            rg,
            gamma,
            rho_u,
        )
        k3u, k3v, k3l, k3x = rhs_python(
            u + 0.5 * h * k2u,
            v + 0.5 * h * k2v,
            l_value + 0.5 * h * k2l,
            sigmat_u,
            c_zv_u,
            cu,
            cp,
            a_value,
            dp,
            rg,
            gamma,
            rho_u,
        )
        k4u, k4v, k4l, k4x = rhs_python(
            u + h * k3u,
            v + h * k3v,
            l_value + h * k3l,
            sigmat_u,
            c_zv_u,
            cu,
            cp,
            a_value,
            dp,
            rg,
            gamma,
            rho_u,
        )

        u += h * (k1u + 2.0 * k2u + 2.0 * k3u + k4u) / 6.0
        v += h * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        l_value += h * (k1l + 2.0 * k2l + 2.0 * k3l + k4l) / 6.0
        x += h * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0

        if not (math.isfinite(u) and math.isfinite(v) and math.isfinite(l_value) and math.isfinite(x)):
            break

        t += h
    return x


def x_end(
    vc: float,
    da: float,
    la: float,
    E_p: float,
    mu_p: float,
    lam_p: float,
    E_u: float,
    sigmat_u: float,
    sigmat_p: float,
    rho_u: float,
    rho_p: float,
    a_u: float,
    l_u: float,
    a_p: float,
    l_p: float,
    max_step: float = 1e-4,
    max_time: float = 1.0,
    max_steps: int = 200_000,
) -> float:
    native_x_end: NativeXEndFn | None = _native_x_end
    if native_x_end is not None:
        return native_x_end(
            vc,
            da,
            la,
            E_p,
            mu_p,
            lam_p,
            E_u,
            sigmat_u,
            sigmat_p,
            rho_u,
            rho_p,
            a_u,
            l_u,
            a_p,
            l_p,
            max_step,
            max_time,
            int(max_steps),
        )
    return x_end_python(
        vc,
        da,
        la,
        E_p,
        mu_p,
        lam_p,
        E_u,
        sigmat_u,
        sigmat_p,
        rho_u,
        rho_p,
        a_u,
        l_u,
        a_p,
        l_p,
        max_step=max_step,
        max_time=max_time,
        max_steps=max_steps,
    )


def native_available() -> bool:
    return _native_rhs is not None and _native_x_end is not None
