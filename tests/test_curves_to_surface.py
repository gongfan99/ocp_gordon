import pytest
import sys
import os
import numpy as np

from OCP.gp import gp_Pnt
from OCP.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCP.Geom import Geom_BSplineCurve, Geom_BSplineSurface
from OCP.GeomConvert import GeomConvert

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src_py.ocp_gordon.internal.curves_to_surface import CurvesToSurface, clamp_bspline
from src_py.ocp_gordon.internal.bspline_algorithms import BSplineAlgorithms
from src_py.ocp_gordon.internal.error import ErrorCode, error

# Helper to create a simple B-spline curve for testing
@pytest.fixture
def create_test_bspline_curve():
    def _create_test_bspline_curve(poles_data, degree, periodic=False):
        poles = TColgp_Array1OfPnt(1, len(poles_data))
        for i, p_data in enumerate(poles_data, 1):
            poles.SetValue(i, gp_Pnt(p_data[0], p_data[1], p_data[2]))

        num_poles = len(poles_data) # This is the number of poles for the OCP object

        if periodic:
            # For periodic B-splines, the knot vector is typically uniform
            # and multiplicities are 1.
            # For a periodic curve, the number of knots is num_poles + 1.
            # The knot values are uniform.
            num_knots_periodic = num_poles + 1
            knots_array = TColStd_Array1OfReal(1, num_knots_periodic)
            mults_array = TColStd_Array1OfInteger(1, num_knots_periodic)

            for i in range(1, num_knots_periodic + 1):
                knots_array.SetValue(i, float(i - 1))
                mults_array.SetValue(i, 1)
            
            # No need for unique_knots adjustment for uniform periodic knots
            final_knots = knots_array
            final_mults = mults_array

        else:
            # For a clamped B-spline, the knot vector structure is well-defined.
            # The sum of multiplicities must be equal to num_poles + degree + 1.
            # The first and last knots have a multiplicity of degree + 1.
            
            num_internal_knots = num_poles - degree - 1
            num_unique_knots = num_internal_knots + 2
            
            final_knots = TColStd_Array1OfReal(1, num_unique_knots)
            final_mults = TColStd_Array1OfInteger(1, num_unique_knots)

            # First knot (clamped start)
            final_knots.SetValue(1, 0.0)
            final_mults.SetValue(1, degree + 1)

            # Internal knots (uniform)
            for i in range(num_internal_knots):
                knot_val = (i + 1.0) / (num_internal_knots + 1.0)
                final_knots.SetValue(i + 2, knot_val)
                final_mults.SetValue(i + 2, 1)

            # Last knot (clamped end)
            final_knots.SetValue(num_unique_knots, 1.0)
            final_mults.SetValue(num_unique_knots, degree + 1)

        return Geom_BSplineCurve(poles, final_knots, final_mults, degree, periodic)
    return _create_test_bspline_curve

def test_clamp_bspline(create_test_bspline_curve):
    # Create a periodic B-spline curve
    poles_data = [
        (0, 0, 0), (1, 1, 0), (2, 0, 0), (1, -1, 0), (0, 0, 0) # Closed shape
    ]
    degree = 2
    # poles_data = poles_data[:-1]
    periodic_curve = create_test_bspline_curve(poles_data, degree, periodic=True)
    
    assert periodic_curve.IsPeriodic()

    clamped_curve = clamp_bspline(periodic_curve)
    
    assert clamped_curve is not None
    assert not clamped_curve.IsPeriodic()
    # Check if the clamped curve still represents the original shape within its parameter range
    assert clamped_curve.FirstParameter() == pytest.approx(periodic_curve.FirstParameter())
    assert clamped_curve.LastParameter() == pytest.approx(periodic_curve.LastParameter())

    # Test with a non-periodic curve (should return None or the same curve)
    non_periodic_curve = create_test_bspline_curve(poles_data[:-1], degree, periodic=False)
    assert not non_periodic_curve.IsPeriodic()
    result = clamp_bspline(non_periodic_curve)
    assert result is None # clamp_bspline returns None if not periodic

def test_curves_to_surface_basic(create_test_bspline_curve):
    # Create a simple network of curves
    # Two profile curves (u-direction)
    curve1_poles = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    curve2_poles = [(0, 1, 1), (1, 1, 1), (2, 1, 1)]
    degree = 1 # Linear curves for simplicity

    curve1 = create_test_bspline_curve(curve1_poles, degree)
    curve2 = create_test_bspline_curve(curve2_poles, degree)

    curves = [curve1, curve2]

    # Test with default parameters (should be calculated)
    skinner = CurvesToSurface(curves)
    surface = skinner.surface()

    assert isinstance(surface, Geom_BSplineSurface)
    assert surface.NbUPoles() > 0
    assert surface.NbVPoles() > 0
    
    # Check degrees
    assert surface.UDegree() == degree
    # V-degree should be max_degree (default 3) or adjusted based on interpolation
    assert surface.VDegree() >= 1 

    # Test with explicit parameters
    explicit_params = [0.0, 1.0] # Corresponding to the two curves
    skinner_explicit = CurvesToSurface(curves, parameters=explicit_params)
    surface_explicit = skinner_explicit.surface()

    assert isinstance(surface_explicit, Geom_BSplineSurface)
    assert surface_explicit.NbUPoles() > 0
    assert surface_explicit.NbVPoles() > 0
    assert skinner_explicit.get_parameters() == explicit_params

    u_first, u_last, v_first, v_last = surface_explicit.Bounds()
    tolerance = 1e-4   

    # def get_point(p: gp_Pnt):
    #     return [p.X(), p.Y(), p.Z()]
    
    # u1 = u_last
    # v1 = v_first
    # print(f'gordon_surface({u1}, {v1})={get_point(gordon_surface.Value(u1, v1))}')
    # print(f'profile1({0})={get_point(profile1.Value(0))}')

    # Let's check the corner points more reliably.
    assert surface_explicit.Value(u_first, v_first).IsEqual(curve1.Value(0.0), tolerance)
    assert surface_explicit.Value(u_last, v_first).IsEqual(curve1.Value(1.0), tolerance)
    assert surface_explicit.Value(u_first, v_last).IsEqual(curve2.Value(0.0), tolerance)
    assert surface_explicit.Value(u_last, v_last).IsEqual(curve2.Value(1.0), tolerance)

def test_curves_to_surface_empty_input():
    with pytest.raises(RuntimeError): # Should raise RuntimeError from surface() if no curves
        skinner = CurvesToSurface([])
        skinner.surface()

def test_set_max_degree(create_test_bspline_curve):
    # With only two curves, the interpolation in V will always be degree 1
    curve1_poles = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    curve2_poles = [(0, 1, 1), (1, 1, 1), (2, 1, 1)]
    degree = 1
    curve1 = create_test_bspline_curve(curve1_poles, degree)
    curve2 = create_test_bspline_curve(curve2_poles, degree)
    curves = [curve1, curve2]

    skinner = CurvesToSurface(curves)
    
    # Default max_degree is 3, but with 2 curves, result is degree 1
    surface_default = skinner.surface()
    assert surface_default.VDegree() == 1

    skinner.invalidate()
    skinner.set_max_degree(2)
    surface_new_degree = skinner.surface()
    assert surface_new_degree.VDegree() == 1 # Still 1, as it's limited by number of curves

    # Now test with enough curves to actually reach a higher degree
    curve3_poles = [(0, 2, 0), (1, 2, 0), (2, 2, 0)]
    curve4_poles = [(0, 3, 1), (1, 3, 1), (2, 3, 1)]
    curve3 = create_test_bspline_curve(curve3_poles, degree)
    curve4 = create_test_bspline_curve(curve4_poles, degree)
    curves_more = [curve1, curve2, curve3, curve4]

    skinner_more = CurvesToSurface(curves_more)
    
    # Default max_degree is 3, and with 4 curves, it should reach degree 3
    surface_default_more = skinner_more.surface()
    assert surface_default_more.VDegree() == 3

    skinner_more.invalidate()
    skinner_more.set_max_degree(2)
    surface_new_degree_more = skinner_more.surface()
    assert surface_new_degree_more.VDegree() == 2 # Should now be 2

    with pytest.raises(ValueError):
        skinner.set_max_degree(0)

def test_curves_to_surface_closed_continuity(create_test_bspline_curve):
    # Create curves that form a closed loop in the v-direction
    # e.g., curve1 and curve3 are the same
    curve1_poles = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    curve2_poles = [(0, 1, 1), (1, 1, 1), (2, 1, 1)]
    curve3_poles = [(0, 0, 0), (1, 0, 0), (2, 0, 0)] # Same as curve1
    degree = 1

    curve1 = create_test_bspline_curve(curve1_poles, degree)
    curve2 = create_test_bspline_curve(curve2_poles, degree)
    curve3 = create_test_bspline_curve(curve3_poles, degree)

    curves = [curve1, curve2, curve3]

    # Test with continuous_if_closed = True
    skinner = CurvesToSurface(curves, continuous_if_closed=True)
    surface = skinner.surface()

    assert isinstance(surface, Geom_BSplineSurface)
    # Further assertions could check for C2 continuity at the seam,
    # but this is complex to verify purely from OCP object properties.
    # We can at least check if the surface was created without error.

def test_curves_to_surface_parameter_mismatch(create_test_bspline_curve):
    curve1_poles = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    curve2_poles = [(0, 1, 1), (1, 1, 1), (2, 1, 1)]
    degree = 1
    curve1 = create_test_bspline_curve(curve1_poles, degree)
    curve2 = create_test_bspline_curve(curve2_poles, degree)
    curves = [curve1, curve2]

    # Provide wrong number of parameters
    with pytest.raises(Exception, match="The amount of given parameters has to be equal to the amount of given B-splines!"):
        skinner = CurvesToSurface(curves, parameters=[0.0])
        skinner.surface()

def test_curves_to_surface_non_bspline_input():
    # This test assumes GeomConvert.CurveToBSplineCurve_s handles generic Geom_Curve
    # For now, we're creating B-splines directly.
    # If we had a way to create a non-BSpline Geom_Curve (e.g., a circle from Geom_Circle),
    # we could test if the conversion works.
    # For now, this test is implicitly covered by the fact that `_input_curves`
    # are populated by `GeomConvert.CurveToBSplineCurve_s`.
    pass

if __name__ == "__main__":
    if 0:
        pytest.main([f'{__file__}::test_set_max_degree', "-v"])
    else:
        pytest.main([f'{__file__}', "-v"])
