import json
from pathlib import Path
from typing import List

import numpy as np
from OCP.Geom import Geom_BSplineCurve, Geom_BSplineSurface
from OCP.GeomConvert import GeomConvert_CompCurveToBSplineCurve
from OCP.gp import gp_Pnt
from OCP.Precision import Precision
from OCP.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCP.TColStd import (
    TColStd_Array1OfInteger,
    TColStd_Array1OfReal,
    TColStd_Array2OfReal,
)
from scipy.optimize import minimize

# This file implements some missing classes/functions for OCP


class Standard_Real:
    """
    A mutable wrapper for a float value, mimicking C++'s Standard_Real
    when used as an output parameter.
    """

    def __init__(self, value: float = 0.0):
        self._value = float(value)

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, new_value: float):
        self._value = float(new_value)

    def __float__(self) -> float:
        return self._value

    def __repr__(self) -> str:
        return f"Standard_Real({self._value})"

    def __str__(self) -> str:
        return str(self._value)


class math_Vector:
    def __init__(self, lower_index: int, upper_index: int, init_value: float = 0.0):
        if lower_index > upper_index:
            raise ValueError("Lower index cannot be greater than upper index")
        self.lower_index = lower_index
        self.upper_index = upper_index
        self.data = np.full(upper_index - lower_index + 1, init_value, dtype=np.float64)

    def __len__(self):
        return self.upper_index - self.lower_index + 1

    def __call__(self, index: int):
        if not (self.lower_index <= index <= self.upper_index):
            raise IndexError(
                f"Index {index} out of range [{self.lower_index}, {self.upper_index}]"
            )
        return self.data[index - self.lower_index]

    def SetValue(self, index: int, value: float):
        if not (self.lower_index <= index <= self.upper_index):
            raise IndexError(
                f"Index {index} out of range [{self.lower_index}, {self.upper_index}]"
            )
        self.data[index - self.lower_index] = value

    def Value(self, index: int) -> float:
        return self(index)

    def Lower(self) -> int:
        return self.lower_index

    def Upper(self) -> int:
        return self.upper_index

    def Length(self) -> int:
        return len(self)

    def __add__(self, other):
        if not isinstance(other, math_Vector) or len(self) != len(other):
            raise ValueError("Vectors must be of the same dimension for addition")
        result = math_Vector(self.lower_index, self.upper_index)
        result.data = self.data + other.data
        return result

    def __sub__(self, other):
        if not isinstance(other, math_Vector) or len(self) != len(other):
            raise ValueError("Vectors must be of the same dimension for subtraction")
        result = math_Vector(self.lower_index, self.upper_index)
        result.data = self.data - other.data
        return result

    def __mul__(self, scalar: float):
        result = math_Vector(self.lower_index, self.upper_index)
        result.data = self.data * scalar
        return result

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float):
        if scalar == 0:
            raise ValueError("Cannot divide by zero")
        result = math_Vector(self.lower_index, self.upper_index)
        result.data = self.data / scalar
        return result

    def __eq__(self, other):
        if not isinstance(other, math_Vector) or len(self) != len(other):
            return False
        return np.array_equal(self.data, other.data)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"math_Vector(lower={self.lower_index}, upper={self.upper_index}, data={self.data})"


# OCP Geom_BSplineCurve does not have the DownCast() function.
# Hence, to clone a bspline, instead of using Geom_BSplineCurve.DownCast(bspline.Copy()),
# you should use clone_bspline()
def clone_bspline(spline: Geom_BSplineCurve) -> Geom_BSplineCurve:
    """
    Clone a B-spline curve.

    Args:
        spline: Original B-spline curve

    Returns:
        New B-spline
    """
    # Create a copy by manually constructing from existing data
    # Get poles
    poles = TColgp_Array1OfPnt(1, spline.NbPoles())
    spline.Poles(poles)

    # Get weights
    weights = TColStd_Array1OfReal(1, spline.NbPoles())
    spline.Weights(weights)

    # Get knots
    knots = TColStd_Array1OfReal(1, spline.NbKnots())
    spline.Knots(knots)

    # Get multiplicities
    mults = TColStd_Array1OfInteger(1, spline.NbKnots())
    spline.Multiplicities(mults)

    # Create new spline
    new_spline = Geom_BSplineCurve(
        poles, weights, knots, mults, spline.Degree(), spline.IsPeriodic()
    )

    return new_spline


def clone_bspline_surface(surface: Geom_BSplineSurface) -> Geom_BSplineSurface:
    """
    Clone a B-spline surface.

    Args:
        surface: Original B-spline surface

    Returns:
        New B-spline surface
    """
    # Get poles
    poles = TColgp_Array2OfPnt(1, surface.NbUPoles(), 1, surface.NbVPoles())
    surface.Poles(poles)

    # Get weights
    weights = TColStd_Array2OfReal(1, surface.NbUPoles(), 1, surface.NbVPoles())
    surface.Weights(weights)

    # Get U knots
    u_knots = TColStd_Array1OfReal(1, surface.NbUKnots())
    surface.UKnots(u_knots)

    # Get V knots
    v_knots = TColStd_Array1OfReal(1, surface.NbVKnots())
    surface.VKnots(v_knots)

    # Get U multiplicities
    u_mults = TColStd_Array1OfInteger(1, surface.NbUKnots())
    surface.UMultiplicities(u_mults)

    # Get V multiplicities
    v_mults = TColStd_Array1OfInteger(1, surface.NbVKnots())
    surface.VMultiplicities(v_mults)

    # Create new surface
    new_surface = Geom_BSplineSurface(
        poles,
        weights,
        u_knots,
        v_knots,
        u_mults,
        v_mults,
        surface.UDegree(),
        surface.VDegree(),
        surface.IsUPeriodic(),
        surface.IsVPeriodic(),
    )

    return new_surface


class math_MultipleVarFunctionWithGradient:  # placeholder parent class
    def __init__(self):
        pass

    def NbVariables(self) -> int:
        return 2

    def Value(self, X: math_Vector, F: Standard_Real) -> bool:
        G = math_Vector(1, self.NbVariables())
        return self.Values(X, F, G)

    def Gradient(self, X: math_Vector, G: math_Vector) -> bool:
        F = Standard_Real(0.0)
        return self.Values(X, F, G)

    def Values(self, X: math_Vector, F: Standard_Real, G: math_Vector) -> bool:
        return True


# It is difficult to manually implement a stable optimization function.
# Hence scipy.optimize is used.
def math_BFGS(
    aFunc: math_MultipleVarFunctionWithGradient,
    aX: math_Vector,  # Initial guess for X, will be updated with the result
    aTolerance: float,
    aNbIterations: int = 800,
    eZEPS: float = 1.0e-12,
) -> bool:
    nb_variables = aFunc.NbVariables()

    def f(args: list[float]):
        ocp_args = math_Vector(1, nb_variables)
        for i in range(nb_variables):
            ocp_args.SetValue(ocp_args.Lower() + i, args[i])

        F_k = Standard_Real(0.0)
        G_k_vec = math_Vector(1, nb_variables)
        if not aFunc.Values(ocp_args, F_k, G_k_vec):
            raise RuntimeError(f"function cannot evaluate at {args}")
        return F_k.value, np.array([G_k_vec(i) for i in range(1, nb_variables + 1)])

    x0 = np.array([aX(i) for i in range(1, nb_variables + 1)])

    def estimate_initial_inv_hessian(x0, rel_eps=1e-8, reg=1e-8):
        """
        Estimate 2x2 inverse Hessian by finite-differencing the gradient and inverting.
        - x0: 1D array-like, length=2 (current point)
        - grad_func: function(x) -> gradient array-like shape (2,)
        - rel_eps: relative step size for finite differences (default sqrt(machine eps))
        - reg: regularization added to Hessian diagonal before inversion
        Returns: 2x2 numpy array H0 (approx inverse Hessian)
        """
        n = x0.size
        assert n == 2, "This routine expects 2 parameters; adapt if you have more."

        # _, g0 = f(x0)
        H = np.zeros((n, n), dtype=float)

        # central differences for Jacobian of gradient (Hessian)
        for j in range(n):
            # step size scaled to parameter magnitude
            eps = rel_eps * max(1.0, abs(x0[j]))
            ej = np.zeros_like(x0)
            ej[j] = eps
            _, g_plus = f(x0 + ej)
            _, g_minus = f(x0 - ej)
            H[:, j] = (g_plus - g_minus) / (2.0 * eps)

        # symmetrize Hessian (numerical safety)
        H = 0.5 * (H + H.T)

        # regularize: add small multiple of identity to avoid singularity / negative eigenvalues
        # choose reg scale relative to typical diagonal magnitude
        diag_scale = max(np.abs(np.diag(H)).max(), 1.0)
        H_reg = H + np.eye(n) * (reg * diag_scale)

        # Try to invert; if fails, fall back to scaled identity
        try:
            invH = np.linalg.inv(H_reg)
        except np.linalg.LinAlgError:
            # fallback: scaled identity using curvature estimate
            # estimate curvature scale as average diag of |H|
            avg_curv = max(1e-8, np.mean(np.abs(np.diag(H))))
            invH = np.eye(n) * (1.0 / avg_curv)

        # --- Enforce SPD by eigenvalue projection ---
        # Symmetrize
        invH = 0.5 * (invH + invH.T)

        try:
            # Eigen-decompose
            eigvals, eigvecs = np.linalg.eigh(invH)

            # Clamp eigenvalues to a small positive floor
            eigvals_clamped = np.clip(eigvals, rel_eps, None)

            # Reconstruct
            invH_spd = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T

            invH_spd = 0.5 * (invH_spd + invH_spd.T)
            np.linalg.cholesky(invH_spd)
        except np.linalg.LinAlgError:
            print("Hessian projection failed — fallback to identity.")
            avg_curv = max(1e-8, np.mean(np.abs(np.diag(H))))
            invH = np.eye(n) * (1.0 / avg_curv)
            return invH

        return invH_spd

    H0 = estimate_initial_inv_hessian(x0)

    # options = {'gtol': 1e-12, 'ftol': 1e-12, 'maxiter': 100}
    # res = minimize(f, x0=x0, jac=True, method='BFGS', tol=aTolerance, options=options)
    res = minimize(
        f, x0=x0, jac=True, method="BFGS", tol=aTolerance, options={"hess_inv0": H0}
    )

    for i in range(nb_variables):
        aX.SetValue(aX.Lower() + i, res.x[i])

    return res.success


# save_bsplines_to_file() and load_bsplines_from_file() are convenient for debugging
def save_bsplines_to_object(splines: list[Geom_BSplineCurve]):

    obj = []

    for spline in splines:
        poles: list[tuple[float, float, float]] = []
        for i in range(1, spline.NbPoles() + 1):
            p = spline.Pole(i)
            poles.append((p.X(), p.Y(), p.Z()))

        weights = [spline.Weight(i) for i in range(1, spline.NbPoles() + 1)]
        knots = [spline.Knot(i) for i in range(1, spline.NbKnots() + 1)]
        mults = [spline.Multiplicity(i) for i in range(1, spline.NbKnots() + 1)]

        obj.append(
            {
                "poles": poles,
                "weights": weights,
                "knots": knots,
                "mults": mults,
                "degree": spline.Degree(),
                "is_periodic": spline.IsPeriodic(),
                "first_parameter": spline.FirstParameter(),
                "last_parameter": spline.LastParameter(),
            }
        )

    return obj


def save_bsplines_to_file(splines: list[Geom_BSplineCurve], file_path: str):
    with open(Path.home() / file_path, "w") as f:
        json.dump(save_bsplines_to_object(splines), f, indent=2)


def load_bsplines_from_object(objs: list):
    bsplines: list[Geom_BSplineCurve] = []
    for obj in objs:
        # Get poles
        NbPoles = len(obj["poles"])
        poles = TColgp_Array1OfPnt(1, NbPoles)
        for i in range(NbPoles):
            poles.SetValue(i + 1, gp_Pnt(*obj["poles"][i]))

        # Get weights
        weights = TColStd_Array1OfReal(1, NbPoles)
        for i in range(NbPoles):
            weights.SetValue(i + 1, obj["weights"][i])

        # Get knots
        NbKnots = len(obj["knots"])
        knot_array = TColStd_Array1OfReal(1, NbKnots)
        for i in range(NbKnots):
            knot_array.SetValue(i + 1, obj["knots"][i])

        # Get multiplicities
        mult_array = TColStd_Array1OfInteger(1, NbKnots)
        for i in range(NbKnots):
            mult_array.SetValue(i + 1, obj["mults"][i])

        # Create new spline
        new_spline = Geom_BSplineCurve(
            poles, weights, knot_array, mult_array, obj["degree"], obj["is_periodic"]
        )

        if (
            "first_parameter" in obj
            and "last_parameter" in obj
            and (
                obj["first_parameter"] != new_spline.FirstParameter()
                or obj["last_parameter"] != new_spline.LastParameter()
            )
        ):
            new_spline.Segment(obj["first_parameter"], obj["last_parameter"])

        bsplines.append(new_spline)

    return bsplines


def load_bsplines_from_file(file_path: str | Path):

    if Path(file_path).exists():
        full_file_path = Path(file_path)
    else:
        full_file_path = Path.home() / file_path

    with open(full_file_path, "r") as f:
        objs = json.load(f)

    return load_bsplines_from_object(objs)


def concat_two_bsplines(curve1: Geom_BSplineCurve, curve2: Geom_BSplineCurve):
    if curve1.IsClosed() or curve2.IsClosed():
        raise ValueError("The two splines must not be closed")

    max_degree = max(curve1.Degree(), curve2.Degree())
    if curve1.Degree() < max_degree:
        curve1.IncreaseDegree(max_degree)
    if curve2.Degree() < max_degree:
        curve2.IncreaseDegree(max_degree)

    tol = Precision.Confusion_s()

    comb = GeomConvert_CompCurveToBSplineCurve(curve2)
    if not comb.Add(curve1, tol, True, True):
        raise ValueError("The two splines must connect together")
    merged = comb.BSplineCurve()
    return merged
