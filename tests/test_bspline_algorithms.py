import pytest
import sys
import os
import numpy as np
from OCP.Geom import Geom_BSplineCurve, Geom_BSplineSurface
from OCP.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt, TColgp_HArray1OfPnt
from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCP.gp import gp_Pnt
import math

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src_py.ocp_gordon.internal.bspline_algorithms import BSplineAlgorithms, SurfaceDirection
from src_py.ocp_gordon.internal.misc import clone_bspline

# Helper for comparing lists of floats
def assert_list_almost_equal(list1, list2, places=7):
    assert len(list1) == len(list2)
    for i in range(len(list1)):
        assert math.isclose(list1[i], list2[i], rel_tol=10**(-places))

# sum of mults = num_control_points + degree + 1
# first and last values of mults = degree + 1
@pytest.fixture
def simple_bspline_curve():
    poles = TColgp_Array1OfPnt(1, 4)
    poles.SetValue(1, gp_Pnt(0, 0, 0))
    poles.SetValue(2, gp_Pnt(1, 1, 0))
    poles.SetValue(3, gp_Pnt(2, -1, 0))
    poles.SetValue(4, gp_Pnt(3, 0, 0))

    knots = TColStd_Array1OfReal(1, 3)
    knots.SetValue(1, 0.0)
    knots.SetValue(2, 0.5)
    knots.SetValue(3, 1.0)

    mults = TColStd_Array1OfInteger(1, 3)
    mults.SetValue(1, 3)
    mults.SetValue(2, 1)
    mults.SetValue(3, 3)

    return Geom_BSplineCurve(poles, knots, mults, 2) # Degree 2

def test_linspace_with_breaks():
    umin, umax = 0.0, 1.0
    n_values = 5
    breaks = [0.25, 0.75]
    result = BSplineAlgorithms.linspace_with_breaks(umin, umax, n_values, breaks)
    
    assert math.isclose(result[0], 0.0)
    assert math.isclose(result[1], 0.25)
    assert math.isclose(result[2], 0.5)
    assert math.isclose(result[3], 0.75)
    assert math.isclose(result[4], 1.0)
    assert len(result) == 5

    breaks_existing = [0.0, 0.5, 1.0]
    result_existing = BSplineAlgorithms.linspace_with_breaks(umin, umax, n_values, breaks_existing)
    assert len(result_existing) == 5
    assert_list_almost_equal(result_existing, [0.0, 0.25, 0.5, 0.75, 1.0])

    breaks_outside = [-0.1, 1.1]
    result_outside = BSplineAlgorithms.linspace_with_breaks(umin, umax, n_values, breaks_outside)
    assert math.isclose(result_outside[0], -0.1)
    assert math.isclose(result_outside[1], 0.0)
    assert math.isclose(result_outside[-1], 1.1)
    assert len(result_outside) == 7

def test_is_u_dir_closed():
    poles_closed_u = TColgp_Array2OfPnt(1, 2, 1, 2)
    poles_closed_u.SetValue(1, 1, gp_Pnt(0, 0, 0))
    poles_closed_u.SetValue(2, 1, gp_Pnt(0, 1, 0))
    poles_closed_u.SetValue(1, 2, gp_Pnt(1, 0, 0))
    poles_closed_u.SetValue(2, 2, gp_Pnt(1, 1, 0))
    
    poles_closed_u.SetValue(2, 1, gp_Pnt(0, 0, 0))
    poles_closed_u.SetValue(2, 2, gp_Pnt(1, 0, 0))

    assert BSplineAlgorithms.is_u_dir_closed(poles_closed_u, 1e-6)

    poles_not_closed_u = TColgp_Array2OfPnt(1, 2, 1, 2)
    poles_not_closed_u.SetValue(1, 1, gp_Pnt(0, 0, 0))
    poles_not_closed_u.SetValue(2, 1, gp_Pnt(0, 1, 0))
    poles_not_closed_u.SetValue(1, 2, gp_Pnt(1, 0, 0))
    poles_not_closed_u.SetValue(2, 2, gp_Pnt(1, 1, 0))
    assert not BSplineAlgorithms.is_u_dir_closed(poles_not_closed_u, 1e-6)

def test_is_v_dir_closed():
    poles_closed_v = TColgp_Array2OfPnt(1, 2, 1, 2)
    poles_closed_v.SetValue(1, 1, gp_Pnt(0, 0, 0))
    poles_closed_v.SetValue(1, 2, gp_Pnt(1, 0, 0))
    poles_closed_v.SetValue(2, 1, gp_Pnt(0, 1, 0))
    poles_closed_v.SetValue(2, 2, gp_Pnt(1, 1, 0))

    poles_closed_v.SetValue(1, 2, gp_Pnt(0, 0, 0))
    poles_closed_v.SetValue(2, 2, gp_Pnt(0, 1, 0))

    assert BSplineAlgorithms.is_v_dir_closed(poles_closed_v, 1e-6)

    poles_not_closed_v = TColgp_Array2OfPnt(1, 2, 1, 2)
    poles_not_closed_v.SetValue(1, 1, gp_Pnt(0, 0, 0))
    poles_not_closed_v.SetValue(1, 2, gp_Pnt(1, 0, 0))
    poles_not_closed_v.SetValue(2, 1, gp_Pnt(0, 1, 0))
    poles_not_closed_v.SetValue(2, 2, gp_Pnt(1, 1, 0))
    assert not BSplineAlgorithms.is_v_dir_closed(poles_not_closed_v, 1e-6)

def test_compute_params_bspline_curve():
    points_h_array = TColgp_HArray1OfPnt(1, 3)
    points_h_array.SetValue(1, gp_Pnt(0, 0, 0))
    points_h_array.SetValue(2, gp_Pnt(1, 0, 0))
    points_h_array.SetValue(3, gp_Pnt(2, 0, 0))

    params = BSplineAlgorithms.compute_params_bspline_curve(points_h_array, alpha=0.5)
    assert len(params) == 3
    assert math.isclose(params[0], 0.0)
    assert math.isclose(params[1], 0.5)
    assert math.isclose(params[2], 1.0)

    points_h_array_non_uniform = TColgp_HArray1OfPnt(1, 4)
    points_h_array_non_uniform.SetValue(1, gp_Pnt(0, 0, 0))
    points_h_array_non_uniform.SetValue(2, gp_Pnt(1, 0, 0))
    points_h_array_non_uniform.SetValue(3, gp_Pnt(3, 0, 0))
    points_h_array_non_uniform.SetValue(4, gp_Pnt(4, 0, 0))

    params_non_uniform = BSplineAlgorithms.compute_params_bspline_curve(points_h_array_non_uniform, alpha=0.5)
    assert len(params_non_uniform) == 4
    assert math.isclose(params_non_uniform[0], 0.0)
    assert math.isclose(params_non_uniform[1], 1.0 / (1.0 + math.sqrt(2.0) + 1.0))
    assert math.isclose(params_non_uniform[2], (1.0 + math.sqrt(2.0)) / (1.0 + math.sqrt(2.0) + 1.0))
    assert math.isclose(params_non_uniform[3], 1.0)

def test_get_kink_parameters(simple_bspline_curve):
    kinks = BSplineAlgorithms.get_kink_parameters(simple_bspline_curve)
    assert len(kinks) == 0

    poles_kink = TColgp_Array1OfPnt(1, 5)
    poles_kink.SetValue(1, gp_Pnt(0, 0, 0))
    poles_kink.SetValue(2, gp_Pnt(1, 1, 0))
    poles_kink.SetValue(3, gp_Pnt(1, 0, 0))
    poles_kink.SetValue(4, gp_Pnt(2, 1, 0))
    poles_kink.SetValue(5, gp_Pnt(3, 0, 0))

    knots_kink = TColStd_Array1OfReal(1, 3)
    knots_kink.SetValue(1, 0.0)
    knots_kink.SetValue(2, 0.5)
    knots_kink.SetValue(3, 1.0)

    mults_kink = TColStd_Array1OfInteger(1, 3)
    mults_kink.SetValue(1, 3)
    mults_kink.SetValue(2, 2)
    mults_kink.SetValue(3, 3)

    curve_with_kink = Geom_BSplineCurve(poles_kink, knots_kink, mults_kink, 2)

    kinks_found = BSplineAlgorithms.get_kink_parameters(curve_with_kink)
    assert len(kinks_found) == 1
    assert math.isclose(kinks_found[0], 0.5)

def test_trim_curve(simple_bspline_curve):
    trimmed_curve = BSplineAlgorithms.trim_curve(simple_bspline_curve, 0.25, 0.75)
    assert trimmed_curve is not None
    assert math.isclose(trimmed_curve.FirstParameter(), 0.25)
    assert math.isclose(trimmed_curve.LastParameter(), 0.75)
    assert not (trimmed_curve == simple_bspline_curve)

def test_match_parameter_range(simple_bspline_curve):
    curve1 = clone_bspline(simple_bspline_curve)
    curve2_poles = TColgp_Array1OfPnt(1, 4)
    curve2_poles.SetValue(1, gp_Pnt(0, 0, 0))
    curve2_poles.SetValue(2, gp_Pnt(1, 1, 0))
    curve2_poles.SetValue(3, gp_Pnt(2, -1, 0))
    curve2_poles.SetValue(4, gp_Pnt(3, 0, 0))
    curve2_knots = TColStd_Array1OfReal(1, 3)
    curve2_knots.SetValue(1, 0.5)
    curve2_knots.SetValue(2, 1.0)
    curve2_knots.SetValue(3, 1.5)
    curve2_mults = TColStd_Array1OfInteger(1, 3)
    curve2_mults.SetValue(1, 3)
    curve2_mults.SetValue(2, 1)
    curve2_mults.SetValue(3, 3)
    curve2 = Geom_BSplineCurve(curve2_poles, curve2_knots, curve2_mults, 2)

    curves = [curve1, curve2]
    BSplineAlgorithms.match_parameter_range(curves)

    assert math.isclose(curves[0].FirstParameter(), curves[1].FirstParameter())
    assert math.isclose(curves[0].LastParameter(), curves[1].LastParameter())
    assert math.isclose(curves[0].FirstParameter(), simple_bspline_curve.FirstParameter())
    assert math.isclose(curves[0].LastParameter(), simple_bspline_curve.LastParameter())

def test_match_degree(simple_bspline_curve):
    curve1 = clone_bspline(simple_bspline_curve)
    
    poles_deg3 = TColgp_Array1OfPnt(1, 5)
    poles_deg3.SetValue(1, gp_Pnt(0, 0, 0))
    poles_deg3.SetValue(2, gp_Pnt(1, 1, 0))
    poles_deg3.SetValue(3, gp_Pnt(2, -1, 0))
    poles_deg3.SetValue(4, gp_Pnt(3, 1, 0))
    poles_deg3.SetValue(5, gp_Pnt(4, 0, 0))
    knots_deg3 = TColStd_Array1OfReal(1, 3)
    knots_deg3.SetValue(1, 0.0)
    knots_deg3.SetValue(2, 0.5)
    knots_deg3.SetValue(3, 1.0)
    mults_deg3 = TColStd_Array1OfInteger(1, 3)
    mults_deg3.SetValue(1, 4)
    mults_deg3.SetValue(2, 1)
    mults_deg3.SetValue(3, 4)
    curve2 = Geom_BSplineCurve(poles_deg3, knots_deg3, mults_deg3, 3)

    curves = [curve1, curve2]
    BSplineAlgorithms.match_degree(curves)

    assert curves[0].Degree() == 3
    assert curves[1].Degree() == 3

def test_reparametrize_bspline(simple_bspline_curve):
    curve = clone_bspline(simple_bspline_curve)
    BSplineAlgorithms.reparametrize_bspline(curve, 10.0, 20.0)
    assert math.isclose(curve.FirstParameter(), 10.0)
    assert math.isclose(curve.LastParameter(), 20.0)

    original_point_at_0_5 = simple_bspline_curve.Value(0.5)
    reparam_point_at_15_0 = curve.Value(15.0)
    assert original_point_at_0_5.IsEqual(reparam_point_at_15_0, 1e-6)

def test_to_bsplines(simple_bspline_curve):
    curves = [simple_bspline_curve]
    bsplines = BSplineAlgorithms.to_bsplines(curves)
    assert len(bsplines) == 1
    assert isinstance(bsplines[0], Geom_BSplineCurve)

def test_scale_single_curve(simple_bspline_curve):
    scale = BSplineAlgorithms.scale(simple_bspline_curve)
    assert scale > 0.0
    assert scale < 10.0

def test_scale_list_of_curves(simple_bspline_curve):
    curve1 = clone_bspline(simple_bspline_curve)
    curve2_poles = TColgp_Array1OfPnt(1, 2)
    curve2_poles.SetValue(1, gp_Pnt(0, 0, 0))
    curve2_poles.SetValue(2, gp_Pnt(10, 0, 0))
    curve2_knots = TColStd_Array1OfReal(1, 2)
    curve2_knots.SetValue(1, 0.0)
    curve2_knots.SetValue(2, 1.0)
    curve2_mults = TColStd_Array1OfInteger(1, 2)
    curve2_mults.SetValue(1, 2)
    curve2_mults.SetValue(2, 2)
    curve2 = Geom_BSplineCurve(curve2_poles, curve2_knots, curve2_mults, 1)

    curves = [curve1, curve2]
    scale = BSplineAlgorithms.scale(curves)
    assert scale > 0.0
    assert math.isclose(scale, 10.0, rel_tol=0.1)

def test_get_kink_parameters_surface():
    poles_surf = TColgp_Array2OfPnt(1, 2, 1, 2)
    poles_surf.SetValue(1, 1, gp_Pnt(0, 0, 0))
    poles_surf.SetValue(1, 2, gp_Pnt(1, 0, 0))
    poles_surf.SetValue(2, 1, gp_Pnt(0, 1, 0))
    poles_surf.SetValue(2, 2, gp_Pnt(1, 1, 0))

    knots_u = TColStd_Array1OfReal(1, 2)
    knots_u.SetValue(1, 0.0)
    knots_u.SetValue(2, 1.0)
    mults_u = TColStd_Array1OfInteger(1, 2)
    mults_u.SetValue(1, 2)
    mults_u.SetValue(2, 2)

    knots_v = TColStd_Array1OfReal(1, 2)
    knots_v.SetValue(1, 0.0)
    knots_v.SetValue(2, 1.0)
    mults_v = TColStd_Array1OfInteger(1, 2)
    mults_v.SetValue(1, 2)
    mults_v.SetValue(2, 2)

    test_surface = Geom_BSplineSurface(poles_surf, knots_u, knots_v, mults_u, mults_v, 1, 1)

    kinks = BSplineAlgorithms.get_kink_parameters_surface(test_surface)
    assert len(kinks.u) == 0
    assert len(kinks.v) == 0

def test_knots_from_curve_parameters_open():
    params = [0.0, 0.25, 0.5, 0.75, 1.0]
    degree = 2
    knots = BSplineAlgorithms.knots_from_curve_parameters(params, degree, closed_curve=False)
    assert len(knots) == len(params) + degree + 1
    assert_list_almost_equal(knots, [0.0, 0.0, 0.0, 0.375, 0.625, 1.0, 1.0, 1.0])

def test_knots_from_curve_parameters_closed_even_degree():
    params = [0.0, 0.25, 0.5, 0.75, 1.0]
    degree = 2
    params_copy = list(params)
    knots = BSplineAlgorithms.knots_from_curve_parameters(params_copy, degree, closed_curve=True)
    assert len(knots) == 9
    assert knots[0] <= knots[1]
    assert knots[-1] >= knots[-2]

def test_knots_from_curve_parameters_closed_odd_degree():
    params = [0.0, 0.25, 0.5, 0.75, 1.0]
    degree = 3
    params_copy = list(params)
    knots = BSplineAlgorithms.knots_from_curve_parameters(params_copy, degree, closed_curve=True)
    assert len(knots) == 11
    assert knots[0] <= knots[1]
    assert knots[-1] >= knots[-2]

def test_compute_params_bspline_surf():
    poles_surf = TColgp_Array2OfPnt(1, 3, 1, 3)
    poles_surf.SetValue(1, 1, gp_Pnt(0, 0, 0))
    poles_surf.SetValue(1, 2, gp_Pnt(1, 0, 0))
    poles_surf.SetValue(1, 3, gp_Pnt(2, 0, 0))
    poles_surf.SetValue(2, 1, gp_Pnt(0, 1, 0))
    poles_surf.SetValue(2, 2, gp_Pnt(1, 1, 0))
    poles_surf.SetValue(2, 3, gp_Pnt(2, 1, 0))
    poles_surf.SetValue(3, 1, gp_Pnt(0, 2, 0))
    poles_surf.SetValue(3, 2, gp_Pnt(1, 2, 0))
    poles_surf.SetValue(3, 3, gp_Pnt(2, 2, 0))

    params_u, params_v = BSplineAlgorithms.compute_params_bspline_surf(poles_surf, alpha=0.5)

    assert len(params_u) == 3
    assert len(params_v) == 3
    assert_list_almost_equal(params_u, [0.0, 0.5, 1.0])
    assert_list_almost_equal(params_v, [0.0, 0.5, 1.0])

if __name__ == "__main__":
    # pytest.main([f'{__file__}::test_trim_curve', "-v"])
    pytest.main([f'{__file__}', "-v"])
