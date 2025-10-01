import numpy as np
import pytest
import sys
import os
import math

from OCP.gp import gp_Pnt, gp_Vec, gp_XYZ
from OCP.Geom import Geom_Curve, Geom_BSplineCurve
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.GeomAPI import GeomAPI_PointsToBSpline

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src_py.ocp_gordon.internal.misc import math_Vector, Standard_Real, math_MultipleVarFunctionWithGradient, math_BFGS

class CurveCurveDistanceObjectiveArcTan(math_MultipleVarFunctionWithGradient):
    def __init__(self, c1: Geom_Curve, c2: Geom_Curve):
        # Call the parent constructor without arguments, as math_MultipleVarFunction has no __init__
        super().__init__()
        self.m_c1 = c1
        self.m_c2 = c2

    def NbVariables(self) -> int:
        return 2

    def Value(self, X: math_Vector, F: Standard_Real) -> bool:
        G = math_Vector(1, 2)
        self.Values(X, F, G)
        return True # The original C++ returns true, and F is an output parameter

    def Gradient(self, X: math_Vector, G: math_Vector) -> bool: # Gradient also takes G as an output parameter
        F_val = Standard_Real(0.0) # Create a temporary Standard_Real for F
        self.Values(X, F_val, G)
        return True

    # @staticmethod
    # def activate(z: float) -> float:
    #     return 1.5 * math.atan(z) / math.pi + 0.5

    # @staticmethod
    # def d_activate(z: float) -> float:
    #     return 1.5 / (math.pi * (1.0 + z**2))

    @staticmethod
    def activate(z: float) -> float:
        return z #0.6 * math.sin(z) + 0.5

    @staticmethod
    def d_activate(z: float) -> float:
        return 1 #0.6 * math.cos(z)

    def getUParam(self, x0: float) -> float:
        umin = self.m_c1.FirstParameter()
        umax = self.m_c1.LastParameter()
        return self.activate(x0) * (umax - umin) + umin

    def getVParam(self, x1: float) -> float:
        vmin = self.m_c2.FirstParameter()
        vmax = self.m_c2.LastParameter()
        return self.activate(x1) * (vmax - vmin) + vmin

    def d_getUParam(self, x0: float) -> float:
        umin = self.m_c1.FirstParameter()
        umax = self.m_c1.LastParameter()
        return self.d_activate(x0) * (umax - umin)

    def d_getVParam(self, x1: float) -> float:
        vmin = self.m_c2.FirstParameter()
        vmax = self.m_c2.LastParameter()
        return self.d_activate(x1) * (vmax - vmin)

    def Values(self, X: math_Vector, F: Standard_Real, G: math_Vector) -> bool:
        u = self.getUParam(X.Value(1))
        v = self.getVParam(X.Value(2))

        p1 = gp_Pnt(0,0,0)
        p2 = gp_Pnt(0,0,0)
        d1_vec = gp_Vec(0,0,0)
        d2_vec = gp_Vec(0,0,0)

        self.m_c1.D1(u, p1, d1_vec)
        self.m_c2.D1(v, p2, d2_vec)

        diff = gp_Vec(p1.X() - p2.X(), p1.Y() - p2.Y(), p1.Z() - p2.Z())
        F.value = diff.SquareMagnitude()
        
        G.SetValue(1, 2. * diff.Dot(d1_vec) * (self.m_c1.LastParameter() - self.m_c1.FirstParameter()) * self.d_getUParam(X.Value(1)))
        G.SetValue(2, -2. * diff.Dot(d2_vec) * (self.m_c2.LastParameter() - self.m_c2.FirstParameter()) * self.d_getVParam(X.Value(2)))

        return True

class CurveCurveDistanceObjective(math_MultipleVarFunctionWithGradient):
    def __init__(self, c1: Geom_Curve, c2: Geom_Curve):
        # Call the parent constructor without arguments, as math_MultipleVarFunction has no __init__
        super().__init__()
        self.m_c1 = c1
        self.m_c2 = c2

    def NbVariables(self) -> int:
        return 2

    def Value(self, X: math_Vector, F: Standard_Real) -> bool:
        G = math_Vector(1, 2)
        self.Values(X, F, G)
        return True # The original C++ returns true, and F is an output parameter

    def Gradient(self, X: math_Vector, G: math_Vector) -> bool: # Gradient also takes G as an output parameter
        F_val = Standard_Real(0.0) # Create a temporary Standard_Real for F
        self.Values(X, F_val, G)
        return True

    @staticmethod
    def activate(z: float) -> float:
        return 0.5 * (math.sin(z) + 1.)

    @staticmethod
    def d_activate(z: float) -> float:
        return 0.5 * math.cos(z)

    def getUParam(self, x0: float) -> float:
        umin = self.m_c1.FirstParameter()
        umax = self.m_c1.LastParameter()
        return self.activate(x0) * (umax - umin) + umin

    def getVParam(self, x1: float) -> float:
        vmin = self.m_c2.FirstParameter()
        vmax = self.m_c2.LastParameter()
        return self.activate(x1) * (vmax - vmin) + vmin

    def d_getUParam(self, x0: float) -> float:
        umin = self.m_c1.FirstParameter()
        umax = self.m_c1.LastParameter()
        return self.d_activate(x0) * (umax - umin)

    def d_getVParam(self, x1: float) -> float:
        vmin = self.m_c2.FirstParameter()
        vmax = self.m_c2.LastParameter()
        return self.d_activate(x1) * (vmax - vmin)

    def Values(self, X: math_Vector, F: Standard_Real, G: math_Vector) -> bool:
        u = self.getUParam(X.Value(1))
        v = self.getVParam(X.Value(2))

        p1 = gp_Pnt(0,0,0)
        p2 = gp_Pnt(0,0,0)
        d1_vec = gp_Vec(0,0,0)
        d2_vec = gp_Vec(0,0,0)

        self.m_c1.D1(u, p1, d1_vec)
        self.m_c2.D1(v, p2, d2_vec)

        diff = gp_Vec(p1.X() - p2.X(), p1.Y() - p2.Y(), p1.Z() - p2.Z())
        F.value = diff.SquareMagnitude()
        
        G.SetValue(1, 2. * diff.Dot(d1_vec) * (self.m_c1.LastParameter() - self.m_c1.FirstParameter()) * self.d_getUParam(X.Value(1)))
        G.SetValue(2, -2. * diff.Dot(d2_vec) * (self.m_c2.LastParameter() - self.m_c2.FirstParameter()) * self.d_getVParam(X.Value(2)))

        return True
    
class QuadraticFunction(math_MultipleVarFunctionWithGradient):
    def __init__(self):
        super().__init__()
        self._nb_variables = 2

    def NbVariables(self) -> int:
        return self._nb_variables

    def Values(self, X: math_Vector, F: Standard_Real, G: math_Vector) -> bool:
        if X.Length() != self._nb_variables:
            return False

        x = X.Value(1)
        y = X.Value(2)

        # Function value: f(x, y) = x^2 + y^2
        F.value = x**2 + y**2

        # Gradient: grad_f(x, y) = [2x, 2y]
        G.SetValue(1, 2 * x)
        G.SetValue(2, 2 * y)
        
        return True

def create_bspline_curve(points: list[gp_Pnt]):
    """
    Create a B-spline curve from a list of points using GeomAPI_PointsToBSpline.
    
    Args:
        points: List of gp_Pnt points
        
    Returns:
        Handle(Geom_BSplineCurve): Approximated B-spline curve
    """
    # Create a regular array
    n_points = len(points)
    array = TColgp_Array1OfPnt(1, n_points)
    
    # Fill the array with points (indexing starts at 1 in OCP)
    for i, point in enumerate(points, 1):
            array.SetValue(i, point)
    
    # Create the approximator with reasonable defaults
    # Parameters: points, min_degree, max_degree, continuity, tolerance
    approximator = GeomAPI_PointsToBSpline(array)
    return approximator.Curve()

def create_test_curves():
    """
    Create test curves that form a proper intersecting network for Gordon surface interpolation.
    
    Returns:
        Tuple of (profiles, guides) - lists of B-spline curves that properly intersect
    """
    profiles: list[Geom_BSplineCurve] = []
    guides: list[Geom_BSplineCurve] = []
    
    # Define grid parameters
    num_profiles = 3
    num_guides = 4
    u_range = 8.0  # Range in u-direction (profiles)
    v_range = 5.0  # Range in v-direction (guides)
    
    # Create intersection points grid
    # This defines where profiles and guides should intersect
    intersection_points = np.zeros((num_profiles, num_guides, 3))
    
    for i in range(num_profiles):
        for j in range(num_guides):
            # Create a grid of points with some variation for a more interesting surface
            u = i * u_range / (num_profiles - 1) if num_profiles > 1 else 0
            v = j * v_range / (num_guides - 1) if num_guides > 1 else 0
            
            # Add some 3D variation to make the surface more interesting
            z = 0.5 * np.sin(u * 0.5) * np.cos(v * 0.5)
            
            intersection_points[i, j] = [u, v, z]
    
    # Create profile curves (u-direction)
    for i in range(num_profiles):
        points: list[gp_Pnt] = []
        
        # Each profile curve goes through all guide intersection points at this profile index
        for j in range(num_guides):
            x, y, z = intersection_points[i, j]
            points.append(gp_Pnt(x, y, z))
        
        # Create B-spline curve through these points
        bspline_curve = create_bspline_curve(points)
        profiles.append(bspline_curve)
    
    # Create guide curves (v-direction)
    for j in range(num_guides):
        points: list[gp_Pnt] = []
        
        # Each guide curve goes through all profile intersection points at this guide index
        for i in range(num_profiles):
            x, y, z = intersection_points[i, j]
            points.append(gp_Pnt(x, y, z))
        
        # Create B-spline curve through these points
        bspline_curve = create_bspline_curve(points)
        guides.append(bspline_curve)
    
    return profiles, guides

def test_math_bfgs_CurveCurveDistanceObjective():
    """
    Tests the math_BFGS function with test_math_bfgs_CurveCurveDistanceObjective.
    The expected minimum is at (0,0).
    """
    print("\nTesting math_BFGS with test_math_bfgs_CurveCurveDistanceObjective")

    # Initial guess
    initial_x = math_Vector(1, 2)
    initial_x.SetValue(1, 0)
    initial_x.SetValue(2, 0)

    profiles, guides = create_test_curves()

    # curve1 = profiles[0]
    # curve2 = guides[0]
    # curve1.Segment(0, 1.0)
    # curve2.Segment(0, 0.125)

    curve1 = profiles[1]
    curve2 = guides[0]
    curve1.Segment(0, 0.0625)
    curve2.Segment(0.425, 0.5)

    # curve1 = profiles[2]
    # curve2 = guides[0]
    # curve1.Segment(0, 0.125)
    # curve2.Segment(0.875, 1.0)

    # func = CurveCurveDistanceObjective(profiles[0], guides[0])
    func = CurveCurveDistanceObjectiveArcTan(curve1, curve2)

    # Set tolerance and max iterations
    tolerance = 1e-10
    max_iterations = 800

    print(f"Initial X: ({initial_x.Value(1)}, {initial_x.Value(2)})")
    initial_F = Standard_Real(0.0)
    initial_G = math_Vector(1, 2)
    func.Values(initial_x, initial_F, initial_G)
    print(f"Initial F: {initial_F.value}")
    print(f"Initial G: ({initial_G.Value(1)}, {initial_G.Value(2)})")

    # Run FRPR
    success = math_BFGS(func, initial_x, tolerance, max_iterations)

    def get_point(p: gp_Pnt):
        return [p.X(), p.Y(), p.Z()]
    
    u = func.getUParam(initial_x.Value(1))
    v = func.getVParam(initial_x.Value(2))
    print(f'u={u}, v={v}, curve1({u})={get_point(curve1.Value(u))}, curve2({v})={get_point(curve2.Value(v))}')
    # assert False

    final_F = Standard_Real(0.0)
    final_G = math_Vector(1, 2)

    print("\n--- Optimization Results ---")
    if success:
        print("Optimization successful!")
        print(f"Final X: ({initial_x.Value(1)}, {initial_x.Value(2)})")
        final_F = Standard_Real(0.0)
        final_G = math_Vector(1, 2)
        func.Values(initial_x, final_F, final_G)
        print(f"Final F: {final_F.value}")
        print(f"Final G: ({final_G.Value(1)}, {final_G.Value(2)})")
    # Assertions for pytest
    assert success, "Optimization should have succeeded"
    # assert abs(u) < 0.1, f"Final X[1] should be close to 0, got {u}"
    # assert abs(v) < 0.1, f"Final X[2] should be close to 0, got {v}"
    
    if success: # Only assert final F and G if optimization was successful
        assert abs(final_F.value) < tolerance, f"Final F should be close to 0, got {final_F.value}"
        # assert abs(final_G.Value(1)) < tolerance, f"Final G[1] should be close to 0, got {final_G.Value(1)}"
        # assert abs(final_G.Value(2)) < tolerance, f"Final G[2] should be close to 0, got {final_G.Value(2)}"

def test_math_bfgs_quadratic_function():
    """
    Tests the math_BFGS function with a simple quadratic function (f(x,y) = x^2 + y^2).
    The expected minimum is at (0,0).
    """
    print("\nTesting math_BFGS with QuadraticFunction (f(x,y) = x^2 + y^2)")

    # Initial guess
    initial_x = math_Vector(1, 2)
    initial_x.SetValue(1, 3.0)
    initial_x.SetValue(2, 4.0)

    # Create function instance
    quad_func = QuadraticFunction()

    # Set tolerance and max iterations
    tolerance = 1e-6
    max_iterations = 100

    print(f"Initial X: ({initial_x.Value(1)}, {initial_x.Value(2)})")
    initial_F = Standard_Real(0.0)
    initial_G = math_Vector(1, 2)
    quad_func.Values(initial_x, initial_F, initial_G)
    print(f"Initial F: {initial_F.value}")
    print(f"Initial G: ({initial_G.Value(1)}, {initial_G.Value(2)})")

    # Run FRPR
    success = math_BFGS(quad_func, initial_x, tolerance, max_iterations)

    final_F = Standard_Real(0.0)
    final_G = math_Vector(1, 2)

    print("\n--- Optimization Results ---")
    if success:
        print("Optimization successful!")
        print(f"Final X: ({initial_x.Value(1)}, {initial_x.Value(2)})")
        final_F = Standard_Real(0.0)
        final_G = math_Vector(1, 2)
        quad_func.Values(initial_x, final_F, final_G)
        print(f"Final F: {final_F.value}")
        print(f"Final G: ({final_G.Value(1)}, {final_G.Value(2)})")
    # Assertions for pytest
    assert success, "Optimization should have succeeded"
    assert abs(initial_x.Value(1)) < 0.1, f"Final X[1] should be close to 0, got {initial_x.Value(1)}"
    assert abs(initial_x.Value(2)) < 0.1, f"Final X[2] should be close to 0, got {initial_x.Value(2)}"
    
    if success: # Only assert final F and G if optimization was successful
        assert abs(final_F.value) < tolerance, f"Final F should be close to 0, got {final_F.value}"
        # assert abs(final_G.Value(1)) < tolerance, f"Final G[1] should be close to 0, got {final_G.Value(1)}"
        # assert abs(final_G.Value(2)) < tolerance, f"Final G[2] should be close to 0, got {final_G.Value(2)}"


if __name__ == "__main__":
    if 0:
        pytest.main([f'{__file__}::test_math_bfgs_CurveCurveDistanceObjective', "-v"])
    else:
        pytest.main([f'{__file__}', "-v"])
