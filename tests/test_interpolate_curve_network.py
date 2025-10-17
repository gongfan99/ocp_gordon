import os
import sys
from typing import List  # Import List

import numpy as np
import pytest

# Import actual OCP dependencies
from OCP.Geom import Geom_BSplineCurve, Geom_BSplineSurface, Geom_Curve
from OCP.GeomAbs import GeomAbs_Shape
from OCP.GeomAPI import GeomAPI_Interpolate, GeomAPI_PointsToBSpline
from OCP.gp import gp_Pnt
from OCP.Precision import Precision
from OCP.TColgp import TColgp_Array1OfPnt, TColgp_HArray1OfPnt
from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import actual internal dependencies
from src_py.ocp_gordon.internal.bspline_algorithms import BSplineAlgorithms
from src_py.ocp_gordon.internal.curve_network_sorter import CurveNetworkSorter
from src_py.ocp_gordon.internal.error import ErrorCode, error
from src_py.ocp_gordon.internal.gordon_surface_builder import GordonSurfaceBuilder
from src_py.ocp_gordon.internal.interpolate_curve_network import (
    CompatibilityError,
    InterpolateCurveNetwork,
    IntersectionError,
    InvalidInputError,
    SurfaceConstructionError,
    interpolate_curve_network,
)
from src_py.ocp_gordon.internal.intersect_bsplines import IntersectBSplines
from src_py.ocp_gordon.internal.misc import (
    concat_two_bsplines,
    load_bsplines_from_object,
    save_bsplines_to_file,
)


def create_bspline_curve(points: list[gp_Pnt], interpolate=True):
    """
    Create a B-spline curve from a list of points using GeomAPI_Interpolate.

    Args:
        points: List of gp_Pnt points

    Returns:
        Handle(Geom_BSplineCurve): interpolated B-spline curve
    """
    # Create a regular array
    if interpolate:
        n_points = len(points)
        array = TColgp_HArray1OfPnt(1, n_points)

        for i, point in enumerate(points, 1):
            array.SetValue(i, point)

        Interpolator = GeomAPI_Interpolate(array, False, 1e-9)
        Interpolator.Perform()
        return Interpolator.Curve()
    else:
        n_points = len(points)
        array = TColgp_Array1OfPnt(1, n_points)

        for i, point in enumerate(points, 1):
            array.SetValue(i, point)

        # Create approximator with tight tolerance to ensure curve passes through points
        approximator = GeomAPI_PointsToBSpline(
            array, 3, 8, GeomAbs_Shape.GeomAbs_C2, 1e-9
        )
        return approximator.Curve()


# Helper function to create a simple B-spline curve
def create_bspline_curve_from_poles(
    points: list[gp_Pnt], degree: int = 3
) -> Geom_BSplineCurve:
    num_poles = len(points)

    # Control points
    poles_array = TColgp_Array1OfPnt(1, num_poles)
    for i, pnt in enumerate(points, 1):
        poles_array.SetValue(i, pnt)

    if num_poles <= degree:
        raise ValueError(f"Insufficient poles ({num_poles}) for degree ({degree})")

    num_internal_knots = num_poles - degree - 1
    num_unique_knots = (
        num_internal_knots + 2
    )  # Number of unique knots (including 0 and 1)

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

    return Geom_BSplineCurve(
        poles_array, final_knots, final_mults, degree, False
    )  # Not periodic


class TestInterpolateCurveNetwork:

    def setup_method(self):
        # Create simple intersecting curves for testing
        # Curve 1 (U-direction): A straight line from (0,0,0) to (1,0,0)
        self.u_curve1_pts = [gp_Pnt(0, 0, 0), gp_Pnt(1, 0, 0)]
        self.u_curve1 = create_bspline_curve_from_poles(self.u_curve1_pts, degree=1)

        # Curve 2 (U-direction): A straight line from (0,1,0) to (1,1,0)
        self.u_curve2_pts = [gp_Pnt(0, 1, 0), gp_Pnt(1, 1, 0)]
        self.u_curve2 = create_bspline_curve_from_poles(self.u_curve2_pts, degree=1)

        self.ucurves = [self.u_curve1, self.u_curve2]

        # Curve 3 (V-direction): A straight line from (0,0,0) to (0,1,0)
        self.v_curve1_pts = [gp_Pnt(0, 0, 0), gp_Pnt(0, 1, 0)]
        self.v_curve1 = create_bspline_curve_from_poles(self.v_curve1_pts, degree=1)

        # Curve 4 (V-direction): A straight line from (1,0,0) to (1,1,0)
        self.v_curve2_pts = [gp_Pnt(0.5, 0, 0), gp_Pnt(0.5, 1, 0)]
        self.v_curve2 = create_bspline_curve_from_poles(self.v_curve2_pts, degree=1)

        # Curve 5 (V-direction): A straight line from (1,0,0) to (1,1,0)
        self.v_curve3_pts = [gp_Pnt(1, 0, 0), gp_Pnt(1, 1, 0)]
        self.v_curve3 = create_bspline_curve_from_poles(self.v_curve3_pts, degree=1)

        self.vcurves = [self.v_curve1, self.v_curve2, self.v_curve3]

    def test_interpolate_curve_network_function_basic(self):
        # This test will now run the full interpolation pipeline
        # It requires all internal dependencies to function correctly.
        surface = interpolate_curve_network(list(self.ucurves), list(self.vcurves))
        assert isinstance(surface, Geom_BSplineSurface)
        # Further assertions could check surface properties if needed
        # e.g., degree, number of poles, evaluation at intersection points

    def test_interpolate_curve_network_function_3_4(self):
        # Define grid parameters
        data_set = [
            (3, 4, 5.0, 5.0),
            (4, 6, 5.0, 8.0),
            (4, 4, 8.0, 5.0),
            (5, 6, 5.0, 8.0),
            (7, 8, 8.0, 5.0),
            (8, 8, 3.0, 9.0),
            (9, 9, 5.0, 8.0),
        ]
        for num_profiles, num_guides, u_range, v_range in data_set:
            profiles: list[Geom_BSplineCurve] = []
            guides: list[Geom_BSplineCurve] = []

            # Create intersection points grid
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

            # save_bsplines_to_file(profiles, "curve1.json")
            # save_bsplines_to_file(guides, "curve2.json")
            surface = interpolate_curve_network(
                list(profiles), list(guides), tolerance=0.0003
            )
            assert isinstance(surface, Geom_BSplineSurface)

    def test_interpolate_curve_network_function_kink(self):
        objs = [
            {
                "poles": [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.6, 0.0],
                    [4.0, 1.0, 0.0],
                ],
                "weights": [1.0, 1.0, 1.0],
                "knots": [0.0, 1.0],
                "mults": [3, 3],
                "degree": 2,
                "is_periodic": False,
                "first_parameter": 0.0,
                "last_parameter": 1.0,
            },
            {
                "poles": [
                    [4.0, 1.0, 0.0],
                    [6.0, 0.4, 0.0],
                    [8.0, 0.0, 0.0],
                ],
                "weights": [1.0, 1.0, 1.0],
                "knots": [0.0, 1.0],
                "mults": [3, 3],
                "degree": 2,
                "is_periodic": False,
                "first_parameter": 0.0,
                "last_parameter": 1.0,
            },
        ]

        curve1, curve2 = load_bsplines_from_object(objs)
        profile1 = concat_two_bsplines(curve1, curve2)
        profile2 = create_bspline_curve(
            [gp_Pnt(0, 8, 0), gp_Pnt(4, 8, 2), gp_Pnt(8, 8, 0)]
        )
        profiles = [profile1, profile2]
        guides = [
            create_bspline_curve([profile1.StartPoint(), profile2.StartPoint()]),
            create_bspline_curve([profile1.EndPoint(), profile2.EndPoint()]),
        ]

        surface = interpolate_curve_network(
            list(profiles), list(guides), tolerance=1e-6
        )
        assert isinstance(surface, Geom_BSplineSurface)

        guides.append(
            create_bspline_curve([gp_Pnt(4.0, 1.0, 0.0), profile2.Value(0.5)])
        )
        surface = interpolate_curve_network(
            list(profiles), list(guides), tolerance=1e-6
        )
        assert isinstance(surface, Geom_BSplineSurface)

    def test_interpolate_curve_network_function_single_point(self):
        single_point = gp_Pnt(0.5, 0.5, 0)

        profiles = [
            self.u_curve1,
            create_bspline_curve_from_poles([single_point, single_point], degree=1),
            self.u_curve2,
        ]
        guides = [
            create_bspline_curve(
                [self.u_curve1.StartPoint(), single_point, self.u_curve2.StartPoint()]
            ),
            create_bspline_curve(
                [self.u_curve1.EndPoint(), single_point, self.u_curve2.EndPoint()]
            ),
        ]

        with pytest.raises(ValueError):
            surface = interpolate_curve_network(
                list(profiles), list(guides), tolerance=1e-6
            )

        single_point = gp_Pnt(0.5, 0.5, 0)

        profiles = [
            create_bspline_curve_from_poles([single_point, single_point], degree=1),
            self.u_curve1,
            self.u_curve2,
        ]
        guides = [
            create_bspline_curve(
                [single_point, self.u_curve1.StartPoint(), self.u_curve2.StartPoint()]
            ),
            create_bspline_curve(
                [single_point, self.u_curve1.EndPoint(), self.u_curve2.EndPoint()]
            ),
        ]

        surface = interpolate_curve_network(
            list(profiles), list(guides), tolerance=1e-6
        )
        assert isinstance(surface, Geom_BSplineSurface)

    def test_init_invalid_input_less_than_two_profiles(self):
        profiles = [self.u_curve1]
        guides = self.vcurves

        with pytest.raises(
            InvalidInputError, match="There must be at least two profiles"
        ):
            InterpolateCurveNetwork(profiles, guides, 1e-4)

    def test_init_invalid_input_less_than_two_guides(self):
        profiles = self.ucurves
        guides = [self.v_curve1]

        with pytest.raises(
            InvalidInputError, match="There must be at least two guides"
        ):
            InterpolateCurveNetwork(profiles, guides, 1e-4)

    def test_init_duplicate_profiles_removed(self):
        # Create a duplicate of u_curve1
        duplicate_u_curve1 = create_bspline_curve_from_poles(
            self.u_curve1_pts, degree=1
        )
        profiles = [
            self.u_curve1,
            duplicate_u_curve1,
            self.u_curve2,
        ]  # u_curve1 is duplicated
        guides = self.vcurves

        interpolator = InterpolateCurveNetwork(profiles, guides, 1e-4)
        assert len(interpolator.profiles) == 2  # Expect 2 unique profiles

    def test_init_duplicate_guides_removed(self):
        # Create a duplicate of v_curve1
        duplicate_v_curve1 = create_bspline_curve_from_poles(
            self.v_curve1_pts, degree=1
        )
        profiles = self.ucurves
        guides = [
            self.v_curve1,
            duplicate_v_curve1,
            self.v_curve2,
        ]  # v_curve1 is duplicated

        interpolator = InterpolateCurveNetwork(profiles, guides, 1e-4)
        assert len(interpolator.guides) == 2  # Expect 2 unique guides

    def test_init_invalid_input_after_duplicate_removal_profiles(self):
        duplicate_u_curve1 = create_bspline_curve_from_poles(
            self.u_curve1_pts, degree=1
        )
        profiles = [self.u_curve1, duplicate_u_curve1]  # Only one unique profile
        guides = self.vcurves

        with pytest.raises(
            InvalidInputError, match="There must be at least two unique profiles"
        ):
            InterpolateCurveNetwork(profiles, guides, 1e-4)

    def test_init_invalid_input_after_duplicate_removal_guides(self):
        duplicate_v_curve1 = create_bspline_curve_from_poles(
            self.v_curve1_pts, degree=1
        )
        profiles = self.ucurves
        guides = [self.v_curve1, duplicate_v_curve1]  # Only one unique guide

        with pytest.raises(
            InvalidInputError, match="There must be at least two unique guides"
        ):
            InterpolateCurveNetwork(profiles, guides, 1e-4)

    # Note: Testing scenarios like "no intersection" or "multiple intersections"
    # without mocking IntersectBSplines would require carefully crafted B-spline
    # geometries that exhibit these behaviors, which is significantly more complex
    # and might be better suited for integration tests of IntersectBSplines itself.
    # For now, I'll omit these tests as they would require complex geometric setups.

    # Similarly, testing "closed profile and guides" without mocks would require
    # creating actual closed B-splines and ensuring they intersect in a way that
    # triggers the specific error, which is also very complex.

    def test_surface_accessors_after_perform(self):
        interpolator = InterpolateCurveNetwork(self.ucurves, self.vcurves, 1e-4)
        interpolator.perform()  # This will run the full pipeline

        assert isinstance(interpolator.surface(), Geom_BSplineSurface)
        assert isinstance(interpolator.surface_profiles(), Geom_BSplineSurface)
        assert isinstance(interpolator.surface_guides(), Geom_BSplineSurface)
        assert isinstance(interpolator.surface_intersections(), Geom_BSplineSurface)
        assert len(interpolator.parameters_profiles()) == len(self.vcurves)
        assert len(interpolator.parameters_guides()) == len(self.ucurves)

    def test_clamp_function(self):
        interpolator = InterpolateCurveNetwork(self.ucurves, self.vcurves, 1e-4)
        assert interpolator._clamp(5, 0, 10) == 5
        assert interpolator._clamp(-1, 0, 10) == 0
        assert interpolator._clamp(11, 0, 10) == 10
        with pytest.raises(ValueError, match="Minimum may not be larger than maximum"):
            interpolator._clamp(5, 10, 0)

    # Note: Testing _ensure_c2 without mocks would require a surface with specific
    # knot multiplicities to verify the RemoveUKnot/RemoveVKnot calls, which is
    # also very complex to set up with real OCP objects.


@pytest.fixture
def setup_interpolate_data_with_zero_length() -> (
    tuple[list[Geom_BSplineCurve], list[Geom_BSplineCurve]]
):
    # Profiles: zero-length, normal, zero-length
    profiles = [
        create_bspline_curve_from_poles(
            [gp_Pnt(0, 0, 0), gp_Pnt(0, 0, 0)], degree=1
        ),  # Zero-length
        create_bspline_curve([gp_Pnt(0, 1, 0), gp_Pnt(1, 1, 0)]),  # Normal
        create_bspline_curve_from_poles(
            [gp_Pnt(0, 2, 0), gp_Pnt(0, 2, 0)], degree=1
        ),  # Zero-length
    ]
    # Guides: normal, zero-length, normal
    guides = [
        create_bspline_curve([gp_Pnt(0, 0, 0), gp_Pnt(0, 2, 0)]),  # Normal
        create_bspline_curve(
            [gp_Pnt(0, 0, 0), gp_Pnt(0.5, 1, 0), gp_Pnt(0, 2, 0)]
        ),  # Normal
        create_bspline_curve(
            [gp_Pnt(0, 0, 0), gp_Pnt(1, 1, 0), gp_Pnt(0, 2, 0)]
        ),  # Normal
    ]
    return profiles, guides


def test_interpolate_curve_network_with_zero_length_inputs(
    setup_interpolate_data_with_zero_length,
):
    profiles, guides = setup_interpolate_data_with_zero_length
    surface = interpolate_curve_network(list(profiles), list(guides))
    assert isinstance(surface, Geom_BSplineSurface)
    # Further assertions could check surface properties or evaluation at points
    # For example, check if the surface passes through the non-zero-length profile's points
    assert surface.Value(0.5, 0.5).IsEqual(
        gp_Pnt(0.5, 1, 0), 1e-4
    )  # Point on the normal profile


if __name__ == "__main__":
    if 0:
        pytest.main(
            [
                f"{__file__}::TestInterpolateCurveNetwork::test_interpolate_curve_network_function_3_4",
                "-v",
            ]
        )
    else:
        pytest.main([f"{__file__}", "-v"])
