# -*- coding: utf-8 -*-

"""
Faster numba-compiled fit model(s).

It's a little messy to keep a relatively similar API to old FitModel.

TODO include Ella's newer T1 models.

Classes
-------
 - `qdmpy.pl.model.FitModel`
 - `qdmpy.pl.model.ConstStretchedExp`
 - `qdmpy.pl.model.ConstDampedRabi`
 - `qdmpy.pl.model.LinearLorentzians`
 - `qdmpy.pl.model.LinearN15Lorentzians`
 - `qdmpy.pl.model.LinearN14Lorentzians`
 - `qdmpy.pl.model.ConstLorentzians`
 - `qdmpy.pl.model.SkewedLorentzians`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.pl.model.FitModel": True,
    "qdmpy.pl.model.ConstStretchedExp": True,
    "qdmpy.pl.model.ConstDampedRabi": True,
    "qdmpy.pl.model.LinearLorentzians": True,
    "qdmpy.pl.model.LinearN15Lorentzians": True,
    "qdmpy.pl.model.LinearN14Lorentzians": True,
    "qdmpy.pl.model.ConstLorentzians": True,
    "qdmpy.pl.model.SkewedLorentzians": True,
}

# ============================================================================

import numpy as np
import numpy.typing as npt
from numba import njit

# ============================================================================


# =======================================================================================
# =======================================================================================
#
# FitModel Class
#
# =======================================================================================
# =======================================================================================


class FitModel:
    """
    FitModel used to fit to data.
    """

    # =================================

    def __call__(self, param_ar: npt.ArrayLike, sweep_arr: npt.ArrayLike):
        """
        Evaluates fitmodel for given parameter values and sweep (affine) parameter values

        Arguments
        ---------
        param_ar : np array, 1D
            Array of parameters fed into each fitfunc (these are what are fit by sc)
        sweep_arr : np array, 1D or number
            Affine parameter where the fit model is evaluated

        Returns
        -------
        Fit model evaluates at sweep_arr (output is same format as sweep_arr input)
        """
        return self._eval(sweep_arr, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _eval(x: npt.ArrayLike, fit_params: npt.ArrayLike):
        raise NotImplementedError()

    # =================================

    def residuals_scipyfit(
        self,
        param_ar: npt.ArrayLike,
        sweep_arr: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
    ):
        """Evaluates residual: fit model (affine params/sweep_arr) - pl values"""
        # NB: pl_vals unused, but left for compat with FitModel & Hamiltonian
        return self._resid(sweep_arr, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _resid(
        x: npt.ArrayLike, pl_vals: npt.ArrayLike, fit_params: npt.ArrayLike
    ):
        raise NotImplementedError()

    # =================================

    def jacobian_scipyfit(
        self,
        param_ar: npt.ArrayLike,
        sweep_arr: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
    ):
        """Evaluates (analytic) jacobian of fitmodel in format expected by scipy
        least_squares"""
        return self._jac(sweep_arr, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _jac(
        x: npt.ArrayLike, pl_vals: npt.ArrayLike, fit_params: npt.ArrayLike
    ):
        raise NotImplementedError()

    # =================================

    def get_param_defn(self) -> tuple[str, ...]:
        """
        Returns list of parameters in fit_model, note there will be duplicates, and they
        do not have numbers e.g. 'pos_0'. Use `qdmpy.pl.model.FitModel.get_param_odict`
        for that purpose.

        Returns
        -------
        param_defn_ar : tuple
            List of parameter names (param_defn) in fit model.
        """
        raise NotImplementedError()

    # =================================

    def get_param_odict(self) -> dict[str, str]:
        """
        get ordered dict of key: param_key (param_name), val: param_unit for all
        parameters in fit_model

        Returns
        -------
        param_dict : dict
            Dictionary containing key: params, values: units.
        """
        raise NotImplementedError()

    # =================================

    def get_param_unit(self, param_name: str, param_number: int) -> str:
        """Get unit for a given param_key (given by param_name + "_" + param_number)

        Arguments
        ---------
        param_name : str
            Name of parameter, e.g. 'pos' or 'sigma_pos'
        param_number : float or int
            Which parameter to use, e.g. 0 for 'pos_0'

        Returns
        -------
        unit : str
            Unit for that parameter, e.g. "constant" -> "Amplitude (a.u.)""
        """
        if param_name == "residual":
            return "Error: sum( || residual(sweep_params) || ) over affine param (a.u.)"
        if param_name.startswith("sigma_"):
            return self.get_param_odict()[
                param_name[6:] + "_" + str(param_number)
            ]
        return self.get_param_odict()[param_name + "_" + str(param_number)]


# ====================================================================================


class ConstStretchedExp(FitModel):
    @staticmethod
    @njit(fastmath=True)
    def _eval(x: npt.ArrayLike, fit_params: npt.ArrayLike):
        c, charac_exp_t, amp_exp, power_exp = fit_params
        return amp_exp * np.exp(-((x / charac_exp_t) ** power_exp)) + c

    @staticmethod
    @njit(fastmath=True)
    def _resid(
        x: npt.ArrayLike, pl_vals: npt.ArrayLike, fit_params: npt.ArrayLike
    ):
        c, charac_exp_t, amp_exp, power_exp = fit_params
        return (
            amp_exp * np.exp(-((x / charac_exp_t) ** power_exp)) + c - pl_vals
        )

    @staticmethod
    @njit(fastmath=True)
    def _jac(
        x: npt.ArrayLike, pl_vals: npt.ArrayLike, fit_params: npt.ArrayLike
    ):
        c, charac_exp_t, amp_exp, power_exp = fit_params
        j = np.empty((np.shape(x)[0], 4))
        j[:, 0] = 1
        j[:, 1] = (1 / charac_exp_t) * (
            amp_exp
            * power_exp
            * np.exp(-((x / charac_exp_t) ** power_exp))
            * (x / charac_exp_t) ** power_exp
        )
        # just lose the 'a'
        j[:, 2] = np.exp(-((x / charac_exp_t) ** power_exp))
        # a e^(-(x/t)^p) (x/t)^p log(x/t)
        j[:, 3] = (
            -amp_exp
            * np.exp(-((x / charac_exp_t) ** power_exp))
            * (x / charac_exp_t) ** power_exp
            * np.log(x / charac_exp_t)
        )
        return j

    def get_param_defn(self) -> tuple[str, ...]:
        return "constant", "charac_exp_t", "amp_exp", "power_exp"

    def get_param_odict(self) -> dict[str, str]:
        return dict(
            [
                ("constant_0", "Amplitude (a.u.)"),
                ("charac_exp_t_0", "Time (s)"),
                ("amp_exp_0", "Amplitude (a.u.)"),
                ("power_exp_0", "Unitless"),
            ]
        )


# ====================================================================================


class ConstDampedRabi(FitModel):
    @staticmethod
    @njit(fastmath=True)
    def _eval(x: npt.ArrayLike, fit_params: npt.ArrayLike):
        c, omega, pos, amp, tau = fit_params
        return amp * np.exp(-(x / tau)) * np.cos(omega * (x - pos)) + c

    @staticmethod
    @njit(fastmath=True)
    def _resid(
        x: npt.ArrayLike, pl_vals: npt.ArrayLike, fit_params: npt.ArrayLike
    ):
        c, omega, pos, amp, tau = fit_params
        return (
            amp * np.exp(-(x / tau)) * np.cos(omega * (x - pos)) + c - pl_vals
        )

    @staticmethod
    @njit(fastmath=True)
    def _jac(
        x: npt.ArrayLike, pl_vals: npt.ArrayLike, fit_params: npt.ArrayLike
    ):
        c, omega, pos, amp, tau = fit_params
        j = np.empty((np.shape(x)[0], 5))
        j[:, 0] = 1
        j[:, 1] = (
            amp * (pos - x) * np.sin(omega * (x - pos)) * np.exp(-x / tau)
        )  # wrt omega
        j[:, 2] = (amp * omega * np.sin(omega * (x - pos))) * np.exp(
            -x / tau
        )  # wrt pos
        j[:, 3] = np.exp(-x / tau) * np.cos(omega * (x - pos))  # wrt amp
        j[:, 4] = (amp * x * np.cos(omega * (x - pos))) / (
            np.exp(x / tau) * tau**2
        )  # wrt tau
        return j

    def get_param_defn(self) -> tuple[str, ...]:
        return (
            "constant",
            "rabi_freq",
            "rabi_t_offset",
            "rabi_amp",
            "rabi_decay_time",
        )

    def get_param_odict(self) -> dict[str, str]:
        return dict(
            [
                ("constant_0", "Amplitude (a.u.)"),
                ("rabi_freq_0", "Omega (rad/s)"),
                ("rabi_t_offset_0", "Tau_0 (s)"),
                ("rabi_amp_0", "Amp (a.u.)"),
                ("rabi_decay_time_0", "Tau_d (s)"),
            ]
        )


# ====================================================================================


class LinearLorentzians(FitModel):
    def __init__(self, n_lorentzians: int):
        self.n_lorentzians = n_lorentzians
        self.fit_functions = {"linear": 1, "lorentzian": n_lorentzians}

    def __call__(self, param_ar: npt.ArrayLike, sweep_arr: npt.ArrayLike):
        return self._eval(self.n_lorentzians, sweep_arr, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _eval(n: int, x: npt.ArrayLike, fit_params: npt.ArrayLike):
        c = fit_params[0]
        m = fit_params[1]
        val = m * x + c
        for i in range(n):
            fwhm = fit_params[i * 3 + 2]
            pos = fit_params[i * 3 + 3]
            amp = fit_params[i * 3 + 4]
            hwhmsqr = (fwhm**2) / 4
            val += amp * hwhmsqr / ((x - pos) ** 2 + hwhmsqr)
        return val

    def residuals_scipyfit(self, param_ar, sweep_arr, pl_vals):
        """Evaluates residual: fit model (affine params/sweep_arr) - pl values"""
        return self._resid(self.n_lorentzians, sweep_arr, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _resid(
        n: int,
        x: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
        fit_params: npt.ArrayLike,
    ):
        c = fit_params[0]
        m = fit_params[1]
        val = m * x + c
        for i in range(n):
            fwhm = fit_params[i * 3 + 2]
            pos = fit_params[i * 3 + 3]
            amp = fit_params[i * 3 + 4]
            hwhmsqr = (fwhm**2) / 4
            val += amp * hwhmsqr / ((x - pos) ** 2 + hwhmsqr)
        return val - pl_vals

    def jacobian_scipyfit(
        self,
        param_ar: npt.ArrayLike,
        sweep_arr: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
    ):
        """Evaluates (analytic) jacobian of fitmodel in format expected by scipy
        least_squares"""
        return self._jac(self.n_lorentzians, sweep_arr, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _jac(
        n: int,
        x: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
        fit_params: npt.ArrayLike,
    ):
        j = np.empty((np.shape(x)[0], 2 + 3 * n), dtype=np.float64)
        j[:, 0] = 1  # wrt constant
        j[:, 1] = x  # wrt m
        for i in range(n):
            fwhm = fit_params[i * 3 + 2]
            c = fit_params[i * 3 + 3]
            a = fit_params[i * 3 + 4]
            g = fwhm / 2

            j[:, 2 + i * 3] = (a * g * (x - c) ** 2) / (
                (x - c) ** 2 + g**2
            ) ** 2
            j[:, 3 + i * 3] = (2 * a * g**2 * (x - c)) / (
                g**2 + (x - c) ** 2
            ) ** 2
            j[:, 4 + i * 3] = g**2 / ((x - c) ** 2 + g**2)
        return j

    def get_param_defn(self) -> tuple[str, ...]:
        defn = ["c", "m"]
        for i in range(self.n_lorentzians):
            defn += ["fwhm", "pos", "amp"]
        return tuple(defn)

    def get_param_odict(self) -> dict[str, str]:
        defn = [
            ("c_0", "Amplitude (a.u.)"),
            ("m_0", "Amplitude per Freq (a.u.)"),
        ]
        for i in range(self.n_lorentzians):
            defn += [(f"fwhm_{i}", "Freq (MHz)")]
            defn += [(f"pos_{i}", "Freq (MHz)")]
            defn += [(f"amp_{i}", "Amp (a.u.)")]
        return dict(defn)


# ====================================================================================


class LinearN15Lorentzians(FitModel):
    def __init__(self, n_lorentzians: int):
        self.n_lorentzians = n_lorentzians
        self.fit_functions = {"linear": 1, "n15lorentzian": n_lorentzians}

    def __call__(self, param_ar: npt.ArrayLike, sweep_arr: npt.ArrayLike):
        return self._eval(self.n_lorentzians, sweep_arr, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _eval(n: int, x: npt.ArrayLike, fit_params: npt.ArrayLike):
        """N14 has 2 resonances separated by 3.03MHz.
        We keep widths, heights fixed for the pair & split the amp/fwhm between
        the two HF levels."""
        c = fit_params[0]
        m = fit_params[1]
        val = m * x + c
        for i in range(n):
            fwhm = fit_params[i * 3 + 2]
            pos = fit_params[i * 3 + 3]
            amp = fit_params[i * 3 + 4]
            hwhmsqr = ((fwhm / 2.0) ** 2) / 4
            val += (
                0.5 * amp * hwhmsqr / ((x - pos - 1.515) ** 2 + hwhmsqr)
            ) + (0.5 * amp * hwhmsqr / ((x - pos + 1.515) ** 2 + hwhmsqr))
        return val

    def residuals_scipyfit(self, param_ar, sweep_arr, pl_vals):
        """Evaluates residual: fit model (affine params/sweep_arr) - pl values"""
        return self._resid(self.n_lorentzians, sweep_arr, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _resid(
        n: int,
        x: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
        fit_params: npt.ArrayLike,
    ):
        c = fit_params[0]
        m = fit_params[1]
        val = m * x + c
        for i in range(n):
            fwhm = fit_params[i * 3 + 2]
            pos = fit_params[i * 3 + 3]
            amp = fit_params[i * 3 + 4]
            hwhmsqr = ((fwhm / 2.0) ** 2) / 4
            val += (
                0.5 * amp * hwhmsqr / ((x - pos - 1.515) ** 2 + hwhmsqr)
            ) + (0.5 * amp * hwhmsqr / ((x - pos + 1.515) ** 2 + hwhmsqr))
        return val - pl_vals

    def jacobian_scipyfit(
        self,
        param_ar: npt.ArrayLike,
        sweep_arr: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
    ):
        """Evaluates (analytic) jacobian of fitmodel in format expected by scipy
        least_squares"""
        return self._jac(self.n_lorentzians, sweep_arr, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _jac(
        n: int,
        x: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
        fit_params: npt.ArrayLike,
    ):
        j = np.empty((np.shape(x)[0], 2 + 3 * n), dtype=np.float64)
        j[:, 0] = 1  # wrt constant
        j[:, 1] = x  # wrt m
        for i in range(n):
            fwhm = fit_params[i * 3 + 2]
            c = fit_params[i * 3 + 3]
            a = fit_params[i * 3 + 4]
            g = (fwhm / 2) / 2

            j[:, 2 + i * 3] = (a * g / 4) * (
                g**2
                * (
                    -1 / (g**2 + (x + 1.515 - c) ** 2) ** 2
                    - 1 / (g**2 + (x - 1.515 - c) ** 2) ** 2
                )
                + 1 / (g**2 + (x + 1.515 - c) ** 2)
                + 1 / (g**2 + (x - 1.515 - c) ** 2)
            )
            j[:, 3 + i * 3] = (
                a
                * g**2
                * (
                    (x - c - 1.515) / ((g**2 + (x - 1.515 - c) ** 2) ** 2)
                    + (x + 1.515 - c) / (g**2 + (x + 1.515 - c) ** 2) ** 2
                )
            )
            j[:, 4 + i * 3] = (g**2 / 2) * (
                1 / (g**2 + (x - c + 1.515) ** 2)
                + 1 / (g**2 + (x - 1.515 - c) ** 2)
            )
        return j

    def get_param_defn(self) -> tuple[str, ...]:
        defn = ["c", "m"]
        for i in range(self.n_lorentzians):
            defn += ["fwhm", "pos", "amp"]
        return tuple(defn)

    def get_param_odict(self) -> dict[str, str]:
        defn = [
            ("c_0", "Amplitude (a.u.)"),
            ("m_0", "Amplitude per Freq (a.u.)"),
        ]
        for i in range(self.n_lorentzians):
            defn += [(f"fwhm_{i}", "Freq (MHz)")]
            defn += [(f"pos_{i}", "Freq (MHz)")]
            defn += [(f"amp_{i}", "Amp (a.u.)")]
        return dict(defn)


# ====================================================================================


class LinearN14Lorentzians(FitModel):
    def __init__(self, n_lorentzians: int):
        self.n_lorentzians = n_lorentzians
        self.fit_functions = {"linear": 1, "n14lorentzian": n_lorentzians}

    def __call__(self, param_ar: npt.ArrayLike, sweep_arr: npt.ArrayLike):
        return self._eval(self.n_lorentzians, sweep_arr, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _eval(n: int, x: npt.ArrayLike, fit_params: npt.ArrayLike):
        """N15 has 3 resonances separated by 2.14MHz.
        We keep widths, heights fixed for the trio & split the amp/fwhm between
        the three HF levels."""
        c = fit_params[0]
        m = fit_params[1]
        val = m * x + c
        for i in range(n):
            fwhm = fit_params[i * 3 + 2]
            pos = fit_params[i * 3 + 3]
            amp = fit_params[i * 3 + 4]
            hwhmsqr = ((fwhm / 3) ** 2) / 4
            val += (
                (amp / 3) * hwhmsqr / ((x - pos - 2.14) ** 2 + hwhmsqr)
                + (amp / 3) * hwhmsqr / ((x - pos) ** 2 + hwhmsqr)
                + (amp / 3) * hwhmsqr / ((x - pos + 2.14) ** 2 + hwhmsqr)
            )
        return val

    def residuals_scipyfit(self, param_ar, sweep_arr, pl_vals):
        """Evaluates residual: fit model (affine params/sweep_arr) - pl values"""
        return self._resid(self.n_lorentzians, sweep_arr, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _resid(
        n: int,
        x: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
        fit_params: npt.ArrayLike,
    ):
        c = fit_params[0]
        m = fit_params[1]
        val = m * x + c
        for i in range(n):
            fwhm = fit_params[i * 3 + 2]
            pos = fit_params[i * 3 + 3]
            amp = fit_params[i * 3 + 4]
            hwhmsqr = ((fwhm / 3) ** 2) / 4
            val += (
                (amp / 3) * hwhmsqr / ((x - pos - 2.14) ** 2 + hwhmsqr)
                + (amp / 3) * hwhmsqr / ((x - pos) ** 2 + hwhmsqr)
                + (amp / 3) * hwhmsqr / ((x - pos + 2.14) ** 2 + hwhmsqr)
            )
        return val - pl_vals

    def jacobian_scipyfit(
        self,
        param_ar: npt.ArrayLike,
        sweep_arr: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
    ):
        """Evaluates (analytic) jacobian of fitmodel in format expected by scipy
        least_squares"""
        return self._jac(self.n_lorentzians, sweep_arr, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _jac(
        n: int,
        x: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
        fit_params: npt.ArrayLike,
    ):
        j = np.empty((np.shape(x)[0], 2 + 3 * n), dtype=np.float64)
        j[:, 0] = 1  # wrt constant
        j[:, 1] = x  # wrt m
        for i in range(n):
            fwhm = fit_params[i * 3 + 2]
            c = fit_params[i * 3 + 3]
            a = fit_params[i * 3 + 4]
            g = (fwhm / 3) / 2

            j[:, 2 + i * 3] = (a * g / 9) * (
                g**2
                * (
                    -1 / (g**2 + (x - c) ** 2) ** 2
                    - 1 / (g**2 + (x + 2.14 - c) ** 2) ** 2
                    - 1 / (g**2 + (x - c - 2.14) ** 2) ** 2
                )
                + 1 / (g**2 + (x - c) ** 2)
                + 1 / (g**2 + (x + 2.14 - c) ** 2)
                + 1 / (g**2 + (x - 2.14 - c) ** 2)
            )
            j[:, 3 + i * 3] = (
                (2 / 3)
                * a
                * g**2
                * (
                    (x - c - 2.14) / ((g**2 + (x - 2.14 - c) ** 2) ** 2)
                    + (x + 2.14 - c) / (g**2 + (x + 2.14 - c) ** 2) ** 2
                    + (x - c) / (g**2 + (x - c) ** 2) ** 2
                )
            )
            j[:, 4 + i * 3] = (g**2 / 3) * (
                1 / (g**2 + (x - c) ** 2)
                + 1 / (g**2 + (x - c + 2.14) ** 2)
                + 1 / (g**2 + (x - 2.14 - c) ** 2)
            )
        return j

    def get_param_defn(self) -> tuple[str, ...]:
        defn = ["c", "m"]
        for i in range(self.n_lorentzians):
            defn += ["fwhm", "pos", "amp"]
        return tuple(defn)

    def get_param_odict(self) -> dict[str, str]:
        defn = [
            ("c_0", "Amplitude (a.u.)"),
            ("m_0", "Amplitude per Freq (a.u.)"),
        ]
        for i in range(self.n_lorentzians):
            defn += [(f"fwhm_{i}", "Freq (MHz)")]
            defn += [(f"pos_{i}", "Freq (MHz)")]
            defn += [(f"amp_{i}", "Amp (a.u.)")]
        return dict(defn)


# ====================================================================================


class ConstLorentzians(FitModel):
    def __init__(self, n_lorentzians):
        self.n_lorentzians = n_lorentzians
        self.fit_functions = {"constant": 1, "lorentzian": n_lorentzians}

    def __call__(self, param_ar: npt.ArrayLike, sweep_arr: npt.ArrayLike):
        return self._eval(self.n_lorentzians, sweep_arr, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _eval(n: int, x: npt.ArrayLike, fit_params: npt.ArrayLike):
        c = fit_params[0]
        val = c * np.ones(np.shape(x))
        for i in range(n):
            fwhm = fit_params[i * 3 + 1]
            pos = fit_params[i * 3 + 2]
            amp = fit_params[i * 3 + 3]
            hwhmsqr = (fwhm**2) / 4
            val += amp * hwhmsqr / ((x - pos) ** 2 + hwhmsqr)
        return val

    def residuals_scipyfit(
        self,
        param_ar: npt.ArrayLike,
        sweep_arr: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
    ):
        """Evaluates residual: fit model (affine params/sweep_arr) - pl values"""
        return self._resid(self.n_lorentzians, sweep_arr, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _resid(
        n: int,
        x: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
        fit_params: npt.ArrayLike,
    ):
        c = fit_params[0]
        val = c * np.ones(np.shape(x))
        for i in range(n):
            fwhm = fit_params[i * 3 + 1]
            pos = fit_params[i * 3 + 2]
            amp = fit_params[i * 3 + 3]
            hwhmsqr = (fwhm**2) / 4
            val += amp * hwhmsqr / ((x - pos) ** 2 + hwhmsqr)
        return val - pl_vals

    def jacobian_scipyfit(
        self,
        param_ar: npt.ArrayLike,
        sweep_arr: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
    ):
        """Evaluates (analytic) jacobian of fitmodel in format expected by
        scipy least_squares"""
        return self._jac(self.n_lorentzians, sweep_arr, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _jac(
        n: int,
        x: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
        fit_params: npt.ArrayLike,
    ):
        j = np.empty((np.shape(x)[0], 1 + 3 * n), dtype=np.float64)
        j[:, 0] = 1  # wrt constant
        for i in range(n):
            fwhm = fit_params[i * 3 + 1]
            c = fit_params[i * 3 + 2]
            a = fit_params[i * 3 + 3]
            g = fwhm / 2

            j[:, 1 + i * 3] = (a * g * (x - c) ** 2) / (
                (x - c) ** 2 + g**2
            ) ** 2
            j[:, 2 + i * 3] = (2 * a * g**2 * (x - c)) / (
                g**2 + (x - c) ** 2
            ) ** 2
            j[:, 3 + i * 3] = g**2 / ((x - c) ** 2 + g**2)
        return j

    def get_param_defn(self) -> tuple[str, ...]:
        defn = ["constant"]
        for i in range(self.n_lorentzians):
            defn += ["fwhm", "pos", "amp"]
        return tuple(defn)

    def get_param_odict(self) -> dict[str, str]:
        defn = [("constant_0", "Amplitude (a.u.)")]
        for i in range(self.n_lorentzians):
            defn += [(f"fwhm_{i}", "Freq (MHz)")]
            defn += [(f"pos_{i}", "Freq (MHz)")]
            defn += [(f"amp_{i}", "Amp (a.u.)")]
        return dict(defn)


# ====================================================================================


class SkewedLorentzians(FitModel):
    def __init__(self):
        self.fit_functions = {"skewed_lorentzians": 1}

    def __call__(self, param_ar: npt.ArrayLike, sweep_arr: npt.ArrayLike):
        return self._eval(sweep_arr, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _eval(x: npt.ArrayLike, fit_params: npt.ArrayLike):
        c = fit_params[0]
        m = fit_params[1]
        val = m * x + c

        D = fit_params[2]
        split = fit_params[3]
        w_l = fit_params[4]
        w_r = fit_params[5]
        a_l = fit_params[6]
        a_r = fit_params[7]
        skew_l = fit_params[8]
        skew_r = fit_params[9]

        dl = x - D - split / 2
        dr = x + D + split / 2

        val += a_l / (
            1 + (dl**2 / (w_l**2 * (1 + skew_l * np.sign(dl)) ** 2))
        )
        val += a_r / (
            1 + (dr**2 / (w_r**2 * (1 + skew_r * np.sign(dr)) ** 2))
        )
        return val

    def residuals_scipyfit(
        self,
        param_ar: npt.ArrayLike,
        sweep_arr: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
    ):
        """Evaluates residual: fit model (affine params/sweep_arr) - pl values"""
        return self._resid(sweep_arr, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _resid(
        x: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
        fit_params: npt.ArrayLike,
    ):
        c = fit_params[0]
        m = fit_params[1]
        val = m * x + c

        D = fit_params[2]
        split = fit_params[3]
        w_l = fit_params[4]
        w_r = fit_params[5]
        a_l = fit_params[6]
        a_r = fit_params[7]
        skew_l = fit_params[8]
        skew_r = fit_params[9]

        dl = x - D + split / 2
        dr = x - D - split / 2

        val += a_l / (
            1 + (dl**2 / (w_l**2 * (1 + skew_l * np.sign(dl)) ** 2))
        )
        val += a_r / (
            1 + (dr**2 / (w_r**2 * (1 + skew_r * np.sign(dr)) ** 2))
        )
        return val - pl_vals

    def jacobian_scipyfit(
        self,
        param_ar: npt.ArrayLike,
        sweep_arr: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
    ):
        """Evaluates (analytic) jacobian of fitmodel in format expected by
        scipy least_squares"""
        return self._jac(sweep_arr, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _jac(
        x: npt.ArrayLike,
        pl_vals: npt.ArrayLike,
        fit_params: npt.ArrayLike,
    ):
        c = fit_params[0]
        m = fit_params[1]
        val = m * x + c

        D = fit_params[2]
        split = fit_params[3]
        w_l = fit_params[4]
        w_r = fit_params[5]
        a_l = fit_params[6]
        a_r = fit_params[7]
        skew_l = fit_params[8]
        skew_r = fit_params[9]

        dl = x - D + split / 2
        dr = x - D - split / 2

        j = np.empty((np.shape(x)[0], 10), dtype=np.float64)
        j[:, 0] = 1  # wrt constant
        j[:, 1] = x  # wrt m

        # wrt D
        j[:, 2] = -1 / 2 * (
            a_r
            * (2 * D + split - 2 * x)
            * (2 - 2 * skew_r * np.sign(D + split / 2 - x))
        ) / (
            w_r**2
            * (1 + skew_r * np.sign(-D - split / 2 + x)) ** 3
            * (
                1
                + (2 * D + split - 2 * x) ** 2
                / (4 * (w_r + skew_r * w_r * np.sign(-D - split / 2 + x)) ** 2)
            )
            ** 2
        ) - (
            a_l
            * (2 * D - split - 2 * x)
            * (2 + 2 * skew_l * np.sign(-D + split / 2 + x))
        ) / (
            2
            * w_l**2
            * (1 + skew_l * np.sign(-D + split / 2 + x)) ** 3
            * (
                1
                + (-2 * D + split + 2 * x) ** 2
                / (4 * (w_l + skew_l * w_l * np.sign(-D + split / 2 + x)) ** 2)
            )
            ** 2
        )

        # wrt split
        j[:, 3] = (
            -(
                (
                    a_r
                    * (2 * D + split - 2 * x)
                    * (2 - 2 * skew_r * np.sign(+split / 2 - x))
                )
                / (
                    wr**2
                    * (1 + skew_r * np.sign(-D - split / 2 + x)) ** 3
                    * (
                        1
                        + (2 * D + split - 2 * x) ** 2
                        / (
                            4
                            * (
                                w_r
                                + skew_r * w_r * np.sign(-D - split / 2 + x)
                            )
                            ** 2
                        )
                    )
                    ** 2
                )
            )
            + (
                al
                * (2 * D - split - 2 * x)
                * (2 + 2 * skew_l * np.sign(-D + split / 2 + x))
            )
            / (
                wl**2
                * (1 + skew_l * np.sign(-D + split / 2 + x)) ** 3
                * (
                    1
                    + (-2 * D + split + 2 * x) ** 2
                    / (
                        4
                        * (w_l + skew_l * w_l * np.sign(-D + split / 2 + x))
                        ** 2
                    )
                )
                ** 2
            )
        ) / 4

        # wrt w_l
        j[:, 4] = (2 * a_l * (-D + split / 2 + x) ** 2) / (
            w_l**3
            * (1 + skew_l * np.sign(-D + split / 2 + x)) ** 2
            * (
                1
                + (-2 * D + split + 2 * x) ** 2
                / (4 * (w_l + skew_l * w_l * np.sign(-D + split / 2 + x)) ** 2)
            )
            ** 2
        )

        # wrt w_r
        j[:, 5] = (2 * a_r * (D + split / 2 - x) ** 2) / (
            wr**3
            * (1 + skew_r * np.sign(-D - split / 2 + x)) ** 2
            * (
                1
                + (2 * D + split - 2 * x) ** 2
                / (4 * (w_r + skew_r * w_r * np.sign(-D - split / 2 + x)) ** 2)
            )
            ** 2
        )

        # wrt a_l
        j[:, 6] = (
            1
            + (-2 * D + split + 2 * x) ** 2
            / (4 * (w_l + skew_l * w_l * np.sign(-D + split / 2 + x)) ** 2)
        ) ** (-1)

        # wrt a_r
        j[:, 7] = (
            1
            + (2 * D + split - 2 * x) ** 2
            / (4 * (w_r + skew_r * w_r * np.sign(-D - split / 2 + x)) ** 2)
        ) ** (-1)

        # wrt skew_l
        j[:, 8] = (
            2 * a_l * (-D + split / 2 + x) ** 2 * np.sign(-D + split / 2 + x)
        ) / (
            w_l**2
            * (1 + skew_l * np.sign(-D + split / 2 + x)) ** 3
            * (
                1
                + (-2 * D + split + 2 * x) ** 2
                / (4 * (wl + skew_l * w_l * np.sign(-D + split / 2 + x)) ** 2)
            )
            ** 2
        )
        # wrt skew_r
        j[:, 9] = (
            2 * a_r * (D + split / 2 - x) ** 2 * np.sign(-D - split / 2 + x)
        ) / (
            w_r**2
            * (1 + skew_r * np.sign(-D - split / 2 + x)) ** 3
            * (
                1
                + (2 * D + split - 2 * x) ** 2
                / (4 * (w_r + skew_r * w_r * np.sign(-D - split / 2 + x)) ** 2)
            )
            ** 2
        )

        return j

    def get_param_defn(self) -> tuple[str, ...]:
        defn = ["constant"]
        for i in range(self.n_lorentzians):
            defn += [
                "c",
                "m",
                "D",
                "split",
                "w_l",
                "w_r",
                "a_l",
                "a_r",
                "skew_l",
                "skew_r",
            ]
        return tuple(defn)

    def get_param_odict(self) -> dict[str, str]:
        defn = [
            ("c_0", "Amplitude (a.u.)"),
            ("m_0", "Amplitude per Freq (a.u.)"),
            ("D_0", "Freq. (MHz)"),
            ("split_0", "Freq. (MHz)"),
            ("w_l_0", "Freq. (MHz)"),
            ("w_r_0", "Freq. (MHz)"),
            ("a_l_0", "Amplitude (a.u.)"),
            ("a_r_0", "Amplitude (a.u.)"),
            ("skew_l_0", "Freq. (MHz)"),
            ("skew_lr_0", "Freq. (MHz)"),
        ]
        return dict(defn)
