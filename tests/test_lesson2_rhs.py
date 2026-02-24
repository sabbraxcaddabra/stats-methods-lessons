import math
import unittest

from scipy.integrate import solve_ivp

from lesson2_rhs import native_available, rhs, rhs_python, x_end, x_end_python


def _reference_rhs(
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
    s_value = 0.5 * rg * (v / u - 1.0) * (1.0 - 1.0 / gamma**2)
    du = rho_u * s_value

    u_num = (
        sigmat_u * (1.0 + (v - u) / c_zv_u)
        + cu * (v - u) ** 2
        - a_value
        - cp * u**2
    )
    u_denum = dp + du
    u_dot = u_num / u_denum
    if u_dot > 0.0:
        u_dot = 0.0

    v_dot = -sigmat_u * (1.0 + (v - u) / c_zv_u) / (rho_u * (l_value - s_value))
    if v_dot > 0.0:
        v_dot = 0.0

    return (u_dot, v_dot, u - v, u)


class Lesson2RhsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = (
            780.0,
            1300.0,
            0.12,
            1.4e9,
            4550.0,
            8500.0,
            3925.0,
            4.2e9,
            145.0,
            0.0032,
            2.1,
            17000.0,
        )
        self.penetration_case = dict(
            da=12e-3,
            la=160e-3,
            vc=1300.0,
            E_p=207e9,
            mu_p=0.33,
            lam_p=0.997,
            E_u=350e9,
            sigmat_u=1400e6,
            sigmat_p=970e6,
            rho_u=17e3,
            rho_p=7850.0,
            a_u=3.83e3,
            l_u=1.5,
            a_p=4.5e3,
            l_p=1.49,
        )

    def _solve_ivp_x_end(self, case: dict) -> float:
        rho_ratio = case["rho_p"] / case["rho_u"]
        a0 = case["l_u"] - case["l_p"] * rho_ratio
        b0 = 2.0 * case["l_u"] * case["vc"] + case["a_u"] + case["a_p"] * rho_ratio
        c0 = case["a_u"] * case["vc"] + case["l_u"] * case["vc"] * case["vc"]
        u0 = (b0 - math.sqrt(b0 * b0 - 4.0 * a0 * c0)) / (2.0 * a0)

        gamma = (case["E_p"] / (3.0 * (1.0 - case["mu_p"]) * case["sigmat_p"])) ** (1.0 / 3.0)
        a_value = 2.0 * (
            case["sigmat_p"] * math.log(gamma)
            + (2.0 / 3.0)
            * case["E_p"]
            * (
                (1.0 / 18.0)
                * math.pi
                * math.pi
                * (1.0 - case["lam_p"])
                * (1.0 - 1.0 / (gamma ** (1.0 / 3.0))) ** 0.62
                + 0.5 * case["sigmat_p"] / (case["E_p"] * case["lam_p"])
            )
        )
        rg = 0.5 * case["da"] * (1.0 + math.sqrt((2.0 * case["rho_u"] * (case["vc"] - u0) ** 2) / a_value))
        dp = case["rho_p"] * rg * (gamma - 1.0) / (gamma + 1.0)
        cu = 0.5 * case["rho_u"]
        cp = 0.5 * case["rho_p"]
        c_zv_u = math.sqrt(case["E_u"] / case["rho_u"])

        y0 = (u0, case["vc"], case["la"], 0.0)

        def rhs_for_ivp(_t: float, y: list[float]) -> tuple[float, float, float, float]:
            return rhs_python(
                y[0],
                y[1],
                y[2],
                case["sigmat_u"],
                c_zv_u,
                cu,
                cp,
                a_value,
                dp,
                rg,
                gamma,
                case["rho_u"],
            )

        zero_u = lambda _t, y: y[0]
        zero_v = lambda _t, y: y[1]
        zero_l = lambda _t, y: y[2]
        zero_u.terminal = True
        zero_v.terminal = True
        zero_l.terminal = True
        zero_u.direction = -1
        zero_v.direction = -1
        zero_l.direction = -1

        solution = solve_ivp(
            rhs_for_ivp,
            (0.0, 1.0),
            y0,
            events=(zero_u, zero_v, zero_l),
            rtol=1e-6,
            atol=1e-8,
        )
        return float(solution.y[3][-1])

    def test_rhs_python_matches_reference_formula(self) -> None:
        expected = _reference_rhs(*self.args)
        actual = rhs_python(*self.args)
        self.assertEqual(len(actual), 4)
        for expected_value, actual_value in zip(expected, actual):
            self.assertAlmostEqual(actual_value, expected_value, places=12)

    def test_rhs_uses_python_path_without_native_module(self) -> None:
        python_result = rhs_python(*self.args)
        auto_result = rhs(*self.args)
        for expected_value, actual_value in zip(python_result, auto_result):
            self.assertAlmostEqual(actual_value, expected_value, places=12)

    def test_native_path_matches_python_when_available(self) -> None:
        if not native_available():
            self.skipTest("native extension is not installed")
        python_result = rhs_python(*self.args)
        native_result = rhs(*self.args)
        for expected_value, actual_value in zip(python_result, native_result):
            self.assertAlmostEqual(actual_value, expected_value, places=12)

    def test_x_end_python_is_close_to_solve_ivp_reference(self) -> None:
        case = self.penetration_case
        expected_x_end = self._solve_ivp_x_end(case)
        actual_x_end = x_end_python(
            case["vc"],
            case["da"],
            case["la"],
            case["E_p"],
            case["mu_p"],
            case["lam_p"],
            case["E_u"],
            case["sigmat_u"],
            case["sigmat_p"],
            case["rho_u"],
            case["rho_p"],
            case["a_u"],
            case["l_u"],
            case["a_p"],
            case["l_p"],
        )
        self.assertAlmostEqual(actual_x_end, expected_x_end, delta=0.01 * expected_x_end)

    def test_x_end_native_matches_python(self) -> None:
        case = self.penetration_case
        expected = x_end_python(
            case["vc"],
            case["da"],
            case["la"],
            case["E_p"],
            case["mu_p"],
            case["lam_p"],
            case["E_u"],
            case["sigmat_u"],
            case["sigmat_p"],
            case["rho_u"],
            case["rho_p"],
            case["a_u"],
            case["l_u"],
            case["a_p"],
            case["l_p"],
        )
        actual = x_end(
            case["vc"],
            case["da"],
            case["la"],
            case["E_p"],
            case["mu_p"],
            case["lam_p"],
            case["E_u"],
            case["sigmat_u"],
            case["sigmat_p"],
            case["rho_u"],
            case["rho_p"],
            case["a_u"],
            case["l_u"],
            case["a_p"],
            case["l_p"],
        )
        self.assertAlmostEqual(actual, expected, delta=1e-6 * max(1.0, abs(expected)))


if __name__ == "__main__":
    unittest.main()
