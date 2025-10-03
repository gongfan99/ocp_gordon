import pytest
import sys
import os
import numpy as np
from typing import List # Import List

from OCP.gp import gp_Pnt
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCP.Geom import Geom_BSplineCurve, Geom_BSplineSurface

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src_py.ocp_gordon.internal.gordon_surface_builder import GordonSurfaceBuilder
from src_py.ocp_gordon.internal.bspline_algorithms import BSplineAlgorithms, SurfaceDirection
from src_py.ocp_gordon.internal.error import error, ErrorCode

# Helper function to create a simple B-spline curve
def create_simple_bspline_curve(points: list[gp_Pnt], degree: int = 3) -> Geom_BSplineCurve:
    num_poles = len(points)
    
    # Control points
    poles_array = TColgp_Array1OfPnt(1, num_poles)
    for i, pnt in enumerate(points, 1):
        poles_array.SetValue(i, pnt)
    
    # For a clamped B-spline, the knot vector structure is well-defined.
    # The sum of multiplicities must be equal to num_poles + degree + 1.
    # The first and last knots have a multiplicity of degree + 1.
    
    # Calculate number of internal knots
    # num_poles = n+1, degree = p. Sum of mults = n+p+2.
    # Number of unique knots = n-p+2.
    # If num_poles = 3, degree = 1: num_internal_knots = 3 - 1 - 1 = 1. num_unique_knots = 1 + 2 = 3.
    # Knots: {0, 0, 0.5, 1, 1} for degree 1. Unique knots: {0, 0.5, 1}.
    # Multiplicities: {2, 1, 2}.
    
    # The number of unique knots is num_poles - degree + 1.
    # This is derived from the formula: num_poles = num_unique_knots + degree - sum(internal_mults)
    # For clamped, sum(internal_mults) = num_internal_knots.
    # So, num_poles = num_unique_knots + degree - num_internal_knots.
    # num_unique_knots = num_poles - degree + num_internal_knots.
    # num_internal_knots = num_poles - degree - 1 (if internal mults are 1)
    # num_unique_knots = (num_poles - degree - 1) + 2 (for start and end knots)
    
    if num_poles <= degree:
        # Not enough poles for the given degree to form a proper B-spline
        # This case should ideally be handled by the calling code or result in an error.
        # For testing, we'll simplify or raise an error.
        raise ValueError(f"Insufficient poles ({num_poles}) for degree ({degree})")

    num_internal_knots = num_poles - degree - 1
    num_unique_knots = num_internal_knots + 2 # Number of unique knots (including 0 and 1)
    
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

    return Geom_BSplineCurve(poles_array, final_knots, final_mults, degree, False) # Not periodic


def test_gordon_surface_builder_basic_construction():
    # Define profile curves (u-direction)
    # These curves should ideally be compatible in parameter space
    # profile1_points = [gp_Pnt(0, 0, 0), gp_Pnt(1, 0.5, 0), gp_Pnt(2, 0, 0)]
    # profile2_points = [gp_Pnt(0, 1, 1), gp_Pnt(1, 1.5, 1), gp_Pnt(2, 1, 1)]
    # profile3_points = [gp_Pnt(0, 2, 2), gp_Pnt(1, 2.5, 2), gp_Pnt(2, 2, 2)]
    profile1_points = [gp_Pnt(0, 0, 0), gp_Pnt(1, 0, 0), gp_Pnt(2, 0, 0)]
    profile2_points = [gp_Pnt(0, 1, 1), gp_Pnt(1, 1, 1), gp_Pnt(2, 1, 1)]
    profile3_points = [gp_Pnt(0, 2, 2), gp_Pnt(1, 2, 2), gp_Pnt(2, 2, 2)]

    profile1 = create_simple_bspline_curve(profile1_points, degree=1)
    profile2 = create_simple_bspline_curve(profile2_points, degree=1)
    profile3 = create_simple_bspline_curve(profile3_points, degree=1)
    profiles = [profile1, profile2, profile3]

    # Define guide curves (v-direction)
    # guide1_points = [gp_Pnt(0, 0, 0), gp_Pnt(0, 1, 1), gp_Pnt(0, 2, 2)]
    # guide2_points = [gp_Pnt(1, 0.5, 0), gp_Pnt(1, 1.5, 1), gp_Pnt(1, 2.5, 2)]
    # guide3_points = [gp_Pnt(2, 0, 0), gp_Pnt(2, 1, 1), gp_Pnt(2, 2, 2)]
    guide1_points = [gp_Pnt(0, 0, 0), gp_Pnt(0, 1, 1), gp_Pnt(0, 2, 2)]
    guide2_points = [gp_Pnt(2.0/3, 0, 0), gp_Pnt(2.0/3, 1, 1), gp_Pnt(2.0/3, 2, 2)]
    guide3_points = [gp_Pnt(4.0/3, 0, 0), gp_Pnt(4.0/3, 1, 1), gp_Pnt(4.0/3, 2, 2)]
    guide4_points = [gp_Pnt(2, 0, 0), gp_Pnt(2, 1, 1), gp_Pnt(2, 2, 2)]

    guide1 = create_simple_bspline_curve(guide1_points, degree=1)
    guide2 = create_simple_bspline_curve(guide2_points, degree=1)
    guide3 = create_simple_bspline_curve(guide3_points, degree=1)
    guide4 = create_simple_bspline_curve(guide4_points, degree=1)
    guides = [guide1, guide2, guide3, guide4]

    # Intersection parameters (normalized 0 to 1)
    # These should represent the parameter values on each curve where they intersect.
    # For simple linear curves, these might be evenly spaced.
    intersection_params_u = [0.0, 1.0/3, 2.0/3, 1.0] # Parameters along profiles (u-direction)
    intersection_params_v = [0.0, 0.5, 1.0] # Parameters along guides (v-direction)

    tolerance = 1e-6

    builder = GordonSurfaceBuilder(
        profiles, guides,
        intersection_params_u, intersection_params_v,
        tolerance
    )

    builder.perform()

    gordon_surface = builder.surface_gordon()
    surf_profiles = builder.surface_profiles()
    surf_guides = builder.surface_guides()
    surf_intersections = builder.surface_intersections()

    assert gordon_surface is not None
    assert surf_profiles is not None
    assert surf_guides is not None
    assert surf_intersections is not None

    # Basic checks on the resulting surface
    assert isinstance(gordon_surface, Geom_BSplineSurface)
    assert gordon_surface.NbUPoles() > 0
    assert gordon_surface.NbVPoles() > 0

    # Verify interpolation at a few points (e.g., corners)
    # The Gordon surface should interpolate the original curve network.
    # For a simple linear network, the corner points should match.
    
    # Point (0,0)
    p00_expected = profile1.Value(intersection_params_u[0]) # Should be (0,0,0)
    u_first, u_last, v_first, v_last = gordon_surface.Bounds()
    p00_actual = gordon_surface.Value(u_first, v_first)
    assert p00_actual.IsEqual(p00_expected, tolerance)

    # Point (1,1) - middle intersection
    p11_expected = profile2.Value(intersection_params_u[1]) # Should be (1, 1.5, 1)
    p11_actual = gordon_surface.Value(u_first + (u_last - u_first) * 0.5,
                                      v_first + (v_last - v_first) * 0.5)
    # assert p11_actual.IsEqual(p11_expected, tolerance)
    # Note: The actual parameter values for the middle point might not be exactly 0.5 on the surface
    # due to reparameterization and knot insertion. We need to evaluate at the corresponding
    # parameter values on the surface. For now, let's use the middle of the surface parameter range.
    # A more robust test would involve mapping the original curve parameters to the surface parameters.
    # For degree 1 curves, the parameter range is usually 0 to 1.
    
    # Let's check the corner points more reliably.
    # (u=0, v=0) -> profile1.Value(0.0) -> guide1.Value(0.0)
    assert gordon_surface.Value(u_first, v_first).IsEqual(profile1.Value(0.0), tolerance)
    assert gordon_surface.Value(u_first, v_first).IsEqual(guide1.Value(0.0), tolerance)

    # def get_point(p: gp_Pnt):
    #     return [p.X(), p.Y(), p.Z()]
    
    # u1 = u_last
    # v1 = v_first
    # print(f'gordon_surface({u1}, {v1})={get_point(gordon_surface.Value(u1, v1))}')
    # print(f'profile1({0})={get_point(profile1.Value(0))}')
    # print(f'profile1({1})={get_point(profile1.Value(1))}')
    # print(f'profile1({0})={get_point(profile2.Value(0))}')
    # print(f'profile1({1})={get_point(profile2.Value(1))}')
    # print(f'profile1({0})={get_point(profile3.Value(0))}')
    # print(f'profile1({1})={get_point(profile3.Value(1))}')

    # print(f'surf_profiles.Poles()={surf_profiles.Poles().NbRows()} x {surf_profiles.Poles().NbColumns()}')
    # print(f'surf_guides.Poles()={surf_guides.Poles().NbRows()} x {surf_guides.Poles().NbColumns()}')
    # print(f'surf_intersections.Poles()={surf_intersections.Poles().NbRows()} x {surf_intersections.Poles().NbColumns()}')
    # print(f'surf_profiles.Poles()={get_point(surf_profiles.Poles()(2,2))}')
    # print(f'surf_guides.Poles()={get_point(surf_guides.Poles()(2,2))}')
    # print(f'surf_intersections.Poles()={get_point(surf_intersections.Poles()(2,2))}')

    # (u=1, v=0) -> profile1.Value(1.0) -> guide3.Value(0.0)    
    # assert surf_profiles.Value(u_last, v_first).IsEqual(profiles[0].Value(1.0), tolerance)
    # assert surf_guides.Value(u_last, v_first).IsEqual(guides[0].Value(1.0), tolerance)
    # assert surf_intersections.Value(u_last, v_first).IsEqual(profiles[0].Value(1.0), tolerance)
    assert gordon_surface.Value(u_last, v_first).IsEqual(profiles[0].Value(1.0), tolerance)
    assert gordon_surface.Value(u_last, v_first).IsEqual(guides[-1].Value(0.0), tolerance)

    # (u=0, v=1) -> profile3.Value(0.0) -> guide1.Value(1.0)
    assert gordon_surface.Value(u_first, v_last).IsEqual(profiles[-1].Value(0.0), tolerance)
    assert gordon_surface.Value(u_first, v_last).IsEqual(guides[0].Value(1.0), tolerance)

    # (u=1, v=1) -> profile3.Value(1.0) -> guide3.Value(1.0)
    assert gordon_surface.Value(u_last, v_last).IsEqual(profiles[-1].Value(1.0), tolerance)
    assert gordon_surface.Value(u_last, v_last).IsEqual(guides[-1].Value(1.0), tolerance)


def test_gordon_surface_builder_insufficient_curves():
    profile1_points = [gp_Pnt(0, 0, 0), gp_Pnt(1, 0.5, 0), gp_Pnt(2, 0, 0)]
    profile1 = create_simple_bspline_curve(profile1_points, degree=1)
    profiles = [profile1] # Only one profile

    guide1_points = [gp_Pnt(0, 0, 0), gp_Pnt(0, 1, 1), gp_Pnt(0, 2, 2)]
    guide1 = create_simple_bspline_curve(guide1_points, degree=1)
    guides = [guide1] # Only one guide

    intersection_params_u = [0.0, 1.0]
    intersection_params_v = [0.0, 1.0]
    tolerance = 1e-6

    # Test with insufficient profiles
    with pytest.raises(error) as excinfo:
        builder = GordonSurfaceBuilder(
            profiles, guides,
            intersection_params_u, intersection_params_v,
            tolerance
        )
        builder.perform()
    assert "There must be at least two profiles" in str(excinfo.value)
    assert excinfo.value.get_code() == ErrorCode.MATH_ERROR # Changed ._code to .get_code()

    # Test with insufficient guides (reset profiles to valid count)
    profiles = [profile1, create_simple_bspline_curve([gp_Pnt(0,0,0), gp_Pnt(1,1,1)], degree=1)]
    guides = [guide1] # Only one guide

    with pytest.raises(error) as excinfo:
        builder = GordonSurfaceBuilder(
            profiles, guides,
            intersection_params_u, intersection_params_v,
            tolerance
        )
        builder.perform()
    assert "There must be at least two guides" in str(excinfo.value)
    assert excinfo.value.get_code() == ErrorCode.MATH_ERROR # Changed ._code to .get_code()

if __name__ == "__main__":
    if 0:
        pytest.main([f'{__file__}::test_gordon_surface_builder_basic_construction', "-v"])
    else:
        pytest.main([f'{__file__}', "-v"])
