import pytest
import sys
import os
import numpy as np
from OCP.Geom import Geom_BSplineCurve, Geom_Curve
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.gp import gp_Pnt
import math

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src_py.ocp_gordon.internal.bspline_approx_interp import BSplineApproxInterp, ProjectResult
from src_py.ocp_gordon.internal.error import error, ErrorCode
from src_py.ocp_gordon.internal.bspline_algorithms import BSplineAlgorithms

# Helper for comparing lists of floats
def assert_list_almost_equal(list1, list2, places=7):
    assert len(list1) == len(list2)
    for i in range(len(list1)):
        assert math.isclose(list1[i], list2[i], rel_tol=10**(-places))

@pytest.fixture
def simple_points_array():
    points = TColgp_Array1OfPnt(1, 4)
    points.SetValue(1, gp_Pnt(0, 0, 0))
    points.SetValue(2, gp_Pnt(1, 1, 0))
    points.SetValue(3, gp_Pnt(2, -1, 0))
    points.SetValue(4, gp_Pnt(3, 0, 0))
    return points

@pytest.fixture
def linear_points_array():
    points = TColgp_Array1OfPnt(1, 3)
    points.SetValue(1, gp_Pnt(0, 0, 0))
    points.SetValue(2, gp_Pnt(1, 0, 0))
    points.SetValue(3, gp_Pnt(2, 0, 0))
    return points

@pytest.fixture
def closed_points_array():
    points = TColgp_Array1OfPnt(1, 5)
    points.SetValue(1, gp_Pnt(0, 0, 0))
    points.SetValue(2, gp_Pnt(1, 0, 0))
    points.SetValue(3, gp_Pnt(1, 1, 0))
    points.SetValue(4, gp_Pnt(0, 1, 0))
    points.SetValue(5, gp_Pnt(0, 0, 0)) # Closed loop
    return points

class TestBSplineApproxInterp:

    def test_init(self, simple_points_array):
        interp = BSplineApproxInterp(simple_points_array, n_control_points=5, degree=3)
        assert interp.m_pnts.Length() == 4
        assert len(interp.m_index_of_approximated) == 4
        assert len(interp.m_index_of_interpolated) == 0
        assert interp.m_degree == 3
        assert interp.m_ncp == 5
        assert not interp.m_c2_continuous

    def test_interpolate_point(self, simple_points_array):
        interp = BSplineApproxInterp(simple_points_array, n_control_points=5, degree=3)
        interp.interpolate_point(1)
        assert 1 not in interp.m_index_of_approximated
        assert 1 in interp.m_index_of_interpolated
        assert len(interp.m_index_of_kinks) == 0

        interp.interpolate_point(2, with_kink=True)
        assert 2 not in interp.m_index_of_approximated
        assert 2 in interp.m_index_of_interpolated
        assert 2 in interp.m_index_of_kinks

        with pytest.raises(error) as excinfo:
            interp.interpolate_point(1) # Already interpolated
        assert excinfo.value.get_code() == ErrorCode.INDEX_ERROR

    def test_max_distance_of_bounding_box(self, simple_points_array):
        interp = BSplineApproxInterp(simple_points_array, n_control_points=5, degree=3)
        dist = interp._max_distance_of_bounding_box(simple_points_array)
        # Bounding box from (0,-1,0) to (3,1,0)
        expected_dist = gp_Pnt(0,-1,0).Distance(gp_Pnt(3,1,0))
        assert math.isclose(dist, expected_dist, rel_tol=1e-6)

    def test_is_closed(self, simple_points_array, closed_points_array):
        # Not C2 continuous, so should return False even if points are closed
        interp_not_c2 = BSplineApproxInterp(closed_points_array, n_control_points=5, degree=3, continuous_if_closed=False)
        assert not interp_not_c2.is_closed()

        # C2 continuous, and points are closed
        interp_c2 = BSplineApproxInterp(closed_points_array, n_control_points=5, degree=3, continuous_if_closed=True)
        assert interp_c2.is_closed()

        # Not closed points
        interp_open = BSplineApproxInterp(simple_points_array, n_control_points=5, degree=3, continuous_if_closed=True)
        assert not interp_open.is_closed()

    def test_first_and_last_interpolated(self, simple_points_array):
        interp = BSplineApproxInterp(simple_points_array, n_control_points=5, degree=3)
        assert not interp._first_and_last_interpolated()

        interp.interpolate_point(0)
        assert not interp._first_and_last_interpolated()

        interp.interpolate_point(simple_points_array.Length() - 1)
        assert interp._first_and_last_interpolated()

    def test_compute_parameters(self, simple_points_array, linear_points_array):
        interp = BSplineApproxInterp(simple_points_array, n_control_points=5, degree=3)
        params = interp._compute_parameters(0.5)
        assert len(params) == 4
        assert math.isclose(params[0], 0.0)
        assert math.isclose(params[-1], 1.0)
        # Specific values depend on point distances, check for monotonicity
        assert params[0] <= params[1] <= params[2] <= params[3]

        interp_linear = BSplineApproxInterp(linear_points_array, n_control_points=3, degree=1)
        params_linear = interp_linear._compute_parameters(0.5)
        assert_list_almost_equal(params_linear, [0.0, 0.5, 1.0])

    def test_compute_knots(self, simple_points_array):
        interp = BSplineApproxInterp(simple_points_array, n_control_points=5, degree=3)
        params = interp._compute_parameters(0.5)
        knots, mults = interp._compute_knots(interp.m_ncp, params)

        # For a clamped B-spline: sum of mults = ncp + degree + 1
        # ncp = 5, degree = 3. Sum of mults should be 5 + 3 + 1 = 9
        assert sum(mults) == interp.m_ncp + interp.m_degree + 1
        # First and last mult should be degree + 1 = 4
        assert mults[0] == interp.m_degree + 1
        assert mults[-1] == interp.m_degree + 1
        # Number of distinct knots = ncp - degree + 1 = 5 - 3 + 1 = 3
        assert len(knots) == interp.m_ncp - interp.m_degree + 1

        # Test with kinks
        interp_kink = BSplineApproxInterp(simple_points_array, n_control_points=5, degree=3)
        interp_kink.interpolate_point(1, with_kink=True) # Add a kink at index 1
        knots_kink, mults_kink = interp_kink._compute_knots(interp_kink.m_ncp, params)
        
        # The number of knots and mults might increase due to kink insertion
        # The sum of multiplicities should still be ncp + degree + 1, but the structure changes
        # The new _insert_knot_with_multiplicity ensures multiplicity is capped at degree
        assert sum(mults_kink) >= interp_kink.m_ncp + interp_kink.m_degree + 1
        assert mults_kink[0] == interp_kink.m_degree + 1
        assert mults_kink[-1] == interp_kink.m_degree + 1
        # Check if the kink parameter is present with increased multiplicity
        kink_param = params[1]
        found_kink_knot = False
        for i, k in enumerate(knots_kink):
            if math.isclose(k, kink_param, rel_tol=1e-4):
                assert mults_kink[i] >= interp_kink.m_degree # Multiplicity should be at least degree
                found_kink_knot = True
                break
        assert found_kink_knot

    def test_fit_curve_simple(self, linear_points_array):
        interp = BSplineApproxInterp(linear_points_array, n_control_points=3, degree=1)
        result = interp.fit_curve()
        assert result.curve is not None
        assert isinstance(result.curve, Geom_BSplineCurve)
        assert result.error >= 0.0
        # For linear points and degree 1, error should be very small
        assert result.error < 1e-6

        # Check if the curve interpolates the points
        for i in range(linear_points_array.Length()):
            p_expected = linear_points_array(i + 1)
            param = interp._compute_parameters(0.5)[i]
            p_actual = result.curve.Value(param)
            assert p_expected.IsEqual(p_actual, 1e-6)

    def test_fit_curve_optimal_simple(self, linear_points_array):
        interp = BSplineApproxInterp(linear_points_array, n_control_points=3, degree=1)
        result = interp.fit_curve_optimal()
        assert result.curve is not None
        assert isinstance(result.curve, Geom_BSplineCurve)
        assert result.error >= 0.0
        assert result.error < 1e-6

        # Check if the curve interpolates the points
        for i in range(linear_points_array.Length()):
            p_expected = linear_points_array(i + 1)
            # Optimal parameters might be slightly different, but should still interpolate
            # For linear, they should be the same as computed
            param = interp._compute_parameters(0.5)[i] # Re-compute for comparison
            p_actual = result.curve.Value(param)
            assert p_expected.IsEqual(p_actual, 1e-6)

    def test_fit_curve_with_interpolation(self, simple_points_array):
        interp = BSplineApproxInterp(simple_points_array, n_control_points=4, degree=3)
        interp.interpolate_point(0) # Interpolate first point
        interp.interpolate_point(3) # Interpolate last point
        
        result = interp.fit_curve()
        assert result.curve is not None
        assert result.error >= 0.0

        # Check if interpolated points are exact
        params = interp._compute_parameters(0.5)
        p0_expected = simple_points_array(1)
        p0_actual = result.curve.Value(params[0])
        assert p0_expected.IsEqual(p0_actual, 1e-6)

        p3_expected = simple_points_array(4)
        p3_actual = result.curve.Value(params[3])
        assert p3_expected.IsEqual(p3_actual, 1e-6)

    def test_fit_curve_with_kinks(self, simple_points_array):
        interp = BSplineApproxInterp(simple_points_array, n_control_points=5, degree=2)
        interp.interpolate_point(1, with_kink=True) # Kink at index 1
        
        result = interp.fit_curve()
        assert result.curve is not None
        assert result.error >= 0.0

        # Check if the kink parameter has high multiplicity in the resulting curve
        params = interp._compute_parameters(0.5)
        kink_param = params[1]
        
        found_kink_mult = False
        for i in range(1, result.curve.NbKnots() + 1):
            if math.isclose(result.curve.Knot(i), kink_param, rel_tol=1e-4):
                assert result.curve.Multiplicity(i) >= interp.m_degree
                found_kink_mult = True
                break
        assert found_kink_mult

    def test_project_on_curve(self, linear_points_array):
        # Create a simple linear B-spline curve
        poles = TColgp_Array1OfPnt(1, 2)
        poles.SetValue(1, gp_Pnt(0, 0, 0))
        poles.SetValue(2, gp_Pnt(2, 0, 0))
        knots = BSplineAlgorithms.to_array([0.0, 1.0])
        mults = BSplineAlgorithms.to_array_int([2, 2]) # Degree 1, clamped
        curve = Geom_BSplineCurve(poles, knots.Array1(), mults.Array1(), 1)

        # Test point on the curve
        pnt_on_curve = gp_Pnt(1, 0, 0)
        initial_param_on = 0.5
        interp = BSplineApproxInterp(linear_points_array, n_control_points=3, degree=1)
        res_on = interp._project_on_curve(pnt_on_curve, curve, initial_param_on)
        assert math.isclose(res_on.parameter, 0.5, rel_tol=1e-6)
        assert res_on.error < 1e-6

        # Test point off the curve
        pnt_off_curve = gp_Pnt(1, 1, 0)
        initial_param_off = 0.5
        res_off = interp._project_on_curve(pnt_off_curve, curve, initial_param_off)
        assert math.isclose(res_off.parameter, 0.5, rel_tol=1e-6) # Closest point is at 0.5
        assert math.isclose(res_off.error, 1.0, rel_tol=1e-6) # Distance to (1,0,0) is 1.0

if __name__ == "__main__":
    # pytest.main([f'{__file__}::TestBSplineApproxInterp::test_fit_curve_with_kinks', "-v"])
    pytest.main([f'{__file__}', "-v"])
