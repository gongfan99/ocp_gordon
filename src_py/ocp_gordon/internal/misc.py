from typing import List
from OCP.Geom import Geom_BSplineCurve, Geom_BSplineSurface
from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger, TColStd_Array2OfReal
from OCP.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
import numpy as np
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
            raise IndexError(f"Index {index} out of range [{self.lower_index}, {self.upper_index}]")
        return self.data[index - self.lower_index]

    def SetValue(self, index: int, value: float):
        if not (self.lower_index <= index <= self.upper_index):
            raise IndexError(f"Index {index} out of range [{self.lower_index}, {self.upper_index}]")
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
    for i in range(1, spline.NbPoles() + 1):
        poles.SetValue(i, spline.Pole(i))
    
    # Get weights
    weights = TColStd_Array1OfReal(1, spline.NbPoles())
    for i in range(1, spline.NbPoles() + 1):
        weights.SetValue(i, spline.Weight(i))
    
    # Get knots
    knot_array = TColStd_Array1OfReal(1, spline.NbKnots())
    for i in range(1, spline.NbKnots() + 1):
        knot_array.SetValue(i, spline.Knot(i))
    
    # Get multiplicities
    mult_array = TColStd_Array1OfInteger(1, spline.NbKnots())
    for i in range(1, spline.NbKnots() + 1):
        mult_array.SetValue(i, spline.Multiplicity(i))
    
    # Create new spline
    new_spline = Geom_BSplineCurve(
        poles, weights, knot_array, mult_array,
        spline.Degree(), spline.IsPeriodic()
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
    for u_idx in range(1, surface.NbUPoles() + 1):
        for v_idx in range(1, surface.NbVPoles() + 1):
            poles.SetValue(u_idx, v_idx, surface.Pole(u_idx, v_idx))
    
    # Get weights
    weights = TColStd_Array2OfReal(1, surface.NbUPoles(), 1, surface.NbVPoles())
    if surface.IsURational() or surface.IsVRational():
        for u_idx in range(1, surface.NbUPoles() + 1):
            for v_idx in range(1, surface.NbVPoles() + 1):
                weights.SetValue(u_idx, v_idx, surface.Weight(u_idx, v_idx))
    else:
        for u_idx in range(1, surface.NbUPoles() + 1):
            for v_idx in range(1, surface.NbVPoles() + 1):
                weights.SetValue(u_idx, v_idx, 1.0) # Non-rational surfaces have weights of 1.0
    
    # Get U knots
    u_knot_array = TColStd_Array1OfReal(1, surface.NbUKnots())
    for i in range(1, surface.NbUKnots() + 1):
        u_knot_array.SetValue(i, surface.UKnot(i))
    
    # Get V knots
    v_knot_array = TColStd_Array1OfReal(1, surface.NbVKnots())
    for i in range(1, surface.NbVKnots() + 1):
        v_knot_array.SetValue(i, surface.VKnot(i))
    
    # Get U multiplicities
    u_mult_array = TColStd_Array1OfInteger(1, surface.NbUKnots())
    for i in range(1, surface.NbUKnots() + 1):
        u_mult_array.SetValue(i, surface.UMultiplicity(i))
    
    # Get V multiplicities
    v_mult_array = TColStd_Array1OfInteger(1, surface.NbVKnots())
    for i in range(1, surface.NbVKnots() + 1):
        v_mult_array.SetValue(i, surface.VMultiplicity(i))
    
    # Create new surface
    new_surface = Geom_BSplineSurface(
        poles, weights, u_knot_array, v_knot_array, u_mult_array, v_mult_array,
        surface.UDegree(), surface.VDegree(), surface.IsUPeriodic(), surface.IsVPeriodic()
    )

    return new_surface


class math_MultipleVarFunctionWithGradient: # placeholder parent class
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
    aX: math_Vector, # Initial guess for X, will be updated with the result
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
        return F_k.value
    
    def g(args: list[float]):
        ocp_args = math_Vector(1, nb_variables)
        for i in range(nb_variables):
            ocp_args.SetValue(ocp_args.Lower() + i, args[i])

        F_k = Standard_Real(0.0)
        G_k_vec = math_Vector(1, nb_variables)
        if not aFunc.Values(ocp_args, F_k, G_k_vec):
            raise RuntimeError(f"gradient cannot evaluate at {args}")
        return np.array([G_k_vec(i) for i in range(1, nb_variables+1)])

    x0 = [aX(i) for i in range(1, nb_variables+1)]
    res = minimize(f, x0=x0, jac=g, method='BFGS', tol=aTolerance)

    if not res.success:
        return False
    
    for i in range(nb_variables):
        aX.SetValue(aX.Lower() + i, res.x[i])
    return True


