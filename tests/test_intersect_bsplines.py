"""
Test cases for intersect_bsplines module.

This module tests the B-spline curve intersection functionality
using recursive subdivision and optimization methods.
"""

import math
import os
import sys

import numpy as np
import pytest
from OCP.Geom import Geom_BSplineCurve, Geom_Circle
from OCP.GeomAbs import GeomAbs_Shape
from OCP.GeomAPI import GeomAPI_PointsToBSpline
from OCP.GeomConvert import GeomConvert
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf
from OCP.TColgp import TColgp_Array1OfPnt

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the module to test
from src_py.ocp_gordon.internal.intersect_bsplines import (
    BoundingBox,
    IntersectBSplines,
    is_point_on_line_segment,
    line_line_intersection_3d,
)
from src_py.ocp_gordon.internal.misc import (
    clone_bspline,
    load_bsplines_from_object,
    save_bsplines_to_file,
)


def create_bspline_from_points(points):
    """
    Create a B-spline curve from a list of (x, y, z) points.

    Args:
        points: List of tuples (x, y, z)

    Returns:
        Geom_BSplineCurve: B-spline curve through the points
    """
    n_points = len(points)
    array = TColgp_Array1OfPnt(1, n_points)

    for i, point in enumerate(points, 1):
        if len(point) == 1:
            array.SetValue(i, gp_Pnt(point[0], 0, 0))
        elif len(point) == 2:
            array.SetValue(i, gp_Pnt(point[0], point[1], 0))
        elif len(point) > 2:
            array.SetValue(i, gp_Pnt(*point))

    # Create approximator with tight tolerance to ensure curve passes through points
    approximator = GeomAPI_PointsToBSpline(array, 3, 8, GeomAbs_Shape.GeomAbs_C2, 1e-9)
    return approximator.Curve()


def create_line_segment(start, end, num_points=10):
    """
    Create a line segment as a B-spline curve.

    Args:
        start: Start point (x, y, z)
        end: End point (x, y, z)
        num_points: Number of points to sample

    Returns:
        Geom_BSplineCurve: B-spline curve representing the line
    """
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        z = start[2] + t * (end[2] - start[2])
        points.append((x, y, z))

    return create_bspline_from_points(points)


class TestIntersectBSplines:
    """Test class for IntersectBSplines functionality."""

    def test_intersect_straight_lines_ends(self):
        """Test intersection of two straight lines that cross."""
        # Create two crossing lines in the XY plane
        line1 = create_line_segment((0, 0, 0), (1, 0, 0))
        line2 = create_line_segment((0, 0, 0), (0, 1, 0))

        intersections = IntersectBSplines(line1, line2, tolerance=1e-6)

        # Should find exactly one intersection at (0, 0, 0)
        assert len(intersections) == 1

        # Check that the intersection point is approximately (0, 0, 0)
        p1 = intersections[0]["point"]
        u = intersections[0]["parmOnCurve1"]
        v = intersections[0]["parmOnCurve2"]
        p2 = line2.Value(
            v
        )  # Re-evaluate p2 using the parameter from the intersection result

        assert abs(p1.X() - 0.0) < 1e-6
        assert abs(p1.Y() - 0.0) < 1e-6
        assert abs(p1.Z() - 0.0) < 1e-6
        assert abs(p2.X() - 0.0) < 1e-6
        assert abs(p2.Y() - 0.0) < 1e-6
        assert abs(p2.Z() - 0.0) < 1e-6

    def test_intersect_straight_lines_crossing(self):
        """Test intersection of two straight lines that cross."""
        # Create two crossing lines in the XY plane
        line1 = create_line_segment((0, 0, 0), (2, 2, 0))
        line2 = create_line_segment((0, 2, 0), (2, 0, 0))

        intersections = IntersectBSplines(line1, line2, tolerance=1e-6)

        # Should find exactly one intersection at (1, 1, 0)
        assert len(intersections) == 1

        # Check that the intersection point is approximately (1, 1, 0)
        p1 = intersections[0]["point"]
        u = intersections[0]["parmOnCurve1"]
        v = intersections[0]["parmOnCurve2"]
        # p1 = line1.Value(u)
        p2 = line2.Value(
            v
        )  # Re-evaluate p2 using the parameter from the intersection result
        # print(f'p1=[{p1.X()}, {p1.Y()}, {p1.Z()}]')
        # print(f'p2=[{p2.X()}, {p2.Y()}, {p2.Z()}]')

        assert abs(p1.X() - 1.0) < 1e-6
        assert abs(p1.Y() - 1.0) < 1e-6
        assert abs(p1.Z() - 0.0) < 1e-6
        assert abs(p2.X() - 1.0) < 1e-6
        assert abs(p2.Y() - 1.0) < 1e-6
        assert abs(p2.Z() - 0.0) < 1e-6

    def test_intersect_straight_lines_end_crossing(self):
        """Test intersection of parallel lines (should not intersect)."""
        line1 = create_line_segment((0, 0, 0), (2, 2, 0))
        line2 = create_line_segment((0, 2, 0), (1, 1, 0))

        intersections = IntersectBSplines(line1, line2, tolerance=1e-6)

        # Should find exactly one intersection at (1, 1, 0)
        assert len(intersections) == 1

        # Check that the intersection point is approximately (1, 1, 0)
        p1 = intersections[0]["point"]
        u = intersections[0]["parmOnCurve1"]
        v = intersections[0]["parmOnCurve2"]
        p2 = line2.Value(
            v
        )  # Re-evaluate p2 using the parameter from the intersection result

        assert abs(p1.X() - 1.0) < 1e-6
        assert abs(p1.Y() - 1.0) < 1e-6
        assert abs(p1.Z() - 0.0) < 1e-6
        assert abs(p2.X() - 1.0) < 1e-6
        assert abs(p2.Y() - 1.0) < 1e-6
        assert abs(p2.Z() - 0.0) < 1e-6

    def test_intersect_straight_lines_not_crossing(self):
        """Test intersection of parallel lines (should not intersect)."""
        line1 = create_line_segment((0, 0, 0), (2, 2, 0))
        line2 = create_line_segment((0, 2, 0), (0.5, 1.5, 0))

        intersections = IntersectBSplines(line1, line2, tolerance=1e-6)

        # lines should not intersect
        assert len(intersections) == 0

    def test_intersect_parallel_lines(self):
        """Test intersection of parallel lines (should not intersect)."""
        line1 = create_line_segment((0, 0, 0), (2, 0, 0))
        line2 = create_line_segment((0, 1, 0), (2, 1, 0))

        intersections = IntersectBSplines(line1, line2, tolerance=1e-6)

        # Parallel lines should not intersect
        assert len(intersections) == 0

    def test_intersect_identical_curves(self):
        """Test intersection of identical curves."""
        # Create a simple curve
        curve = create_bspline_from_points([(0, 0, 0), (1, 1, 0), (2, 0, 0)])

        intersections = IntersectBSplines(curve, curve, tolerance=1e-6)

        # Identical curves should have multiple intersection points
        # (at least one per parameter sample)
        assert len(intersections) > 0

        # All intersection points should be on both curves
        for intersection in intersections:
            u = intersection["parmOnCurve1"]
            v = intersection["parmOnCurve2"]
            p1 = curve.Value(u)
            p2 = curve.Value(v)

            # Points should be very close (within tolerance)
            dx = p2.X() - p1.X()
            dy = p2.Y() - p1.Y()
            dz = p2.Z() - p1.Z()
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            assert distance < 1e-6

    def test_intersect_3d_curves(self):
        """Test intersection of 3D curves."""
        # Create two curves that intersect in 3D space
        # With the improved tolerance in create_bspline_from_points, these should now intersect
        dataset = [
            [(0, 0, 0), (0.1, 0.5, 0), (0, 1, 0)],
            [(0, 0, 0), (1, 0.5, 0), (2, 0.2, 0)],
            [(0, 0, 0), (0.2, 0.2, 0), (0.2, 3, -0.1)],
            [(0, 0, 0), (1, 0.5, 0.2), (2, 0.2, 0.4)],
            [(0, 0, 0), (1, 1, 1), (2, 0, 2)],
            [(0, 2, 2), (1, 1, 1), (2, 2, 0)],
            [(0.2, 0, 0), (0.5, 0.25, 0), (1, 0.4, 0)],
            [(0, 0, 0), (1, 0.5, 0)],
            [(0.2, 0, 0.2), (0.5, 0.25, 0), (1, 0.4, -0.2)],
            [(0, 0, 0), (1, 0.5, 0)],
            [(0.5, 0.26, 0.01), (0.5, 0.25, 0), (0.5, 0.26, -0.01)],
            [(0, 0, 0), (0.5, 0.25, 0), (1, 0.45, 0)],
            [(0.5, 0.25, 0.01), (0.5, 0.25, 0), (0.5, 0.251, -0.01)],
            [(0, 0, 0), (0.5, 0.25, 0), (1, 0.45, 0)],
        ]
        correct_intersec_points = [
            (0, 0, 0),
            (0, 0, 0),
            (1, 1, 1),
            (0.5, 0.25, 0),
            (0.5, 0.25, 0),
            (0.5, 0.25, 0),
            (0.5, 0.25, 0),
        ]
        for i in range(len(correct_intersec_points)):
            curve1 = create_bspline_from_points(dataset[2 * i])
            curve2 = create_bspline_from_points(dataset[2 * i + 1])

            tolerance = 1e-6
            intersections = IntersectBSplines(curve1, curve2, tolerance=tolerance)

            # if len(intersections) == 0:
            #     save_bsplines_to_file([curve1, curve2], "curve1.json")

            # Should find at least one intersection near (1, 1, 1)
            assert len(intersections) >= 1, f"#{i} dataset failed"

            # Check that at least one intersection is near (1, 1, 1)
            found_intersection = False
            for intersection in intersections:
                p = intersection["point"]
                t = gp_Pnt(*correct_intersec_points[i])
                # print(f'distance={p.Distance(t)}')
                if (
                    abs(p.X() - t.X()) < tolerance
                    and abs(p.Y() - t.Y()) < tolerance
                    and abs(p.Z() - t.Z()) < tolerance
                ):
                    found_intersection = True
                    break

            assert found_intersection, f"#{i} dataset failed"

    def test_intersect_tolerance_handling(self):
        """Test that tolerance parameter works correctly."""
        # Create two lines that are very close but don't intersect
        line1 = create_line_segment((0, 0, 0), (2, 0, 0))
        line2 = create_line_segment((0, 1e-7, 0), (2, 1e-7, 0))  # Very close to line1

        # With small tolerance, should not find intersection
        intersections_small_tol = IntersectBSplines(line1, line2, tolerance=1e-8)
        assert len(intersections_small_tol) == 0

        # With larger tolerance, might find intersection
        intersections_large_tol = IntersectBSplines(line1, line2, tolerance=1e-5)
        # This could go either way depending on the algorithm, so we just test it runs

    def test_bbox_intersection_detection(self):
        """Test bounding box intersection detection."""
        # Create curves with known bounding boxes
        curve1 = create_line_segment((0, 0, 0), (1, 1, 1))
        curve2 = create_line_segment((0.5, 0.5, 0.5), (2, 2, 2))

        # Get bounding boxes
        bbox1 = BoundingBox(curve1)
        bbox2 = BoundingBox(curve2)

        # Should intersect
        assert bbox1.Intersects(bbox2, 1e-6)

        # Create non-intersecting curves
        curve3 = create_line_segment((10, 10, 10), (11, 11, 11))
        bbox3 = BoundingBox(curve3)

        # Should not intersect
        assert not bbox1.Intersects(bbox3, 1e-6)

    def test_intersect_straight_lines_parameter_check(self):
        """Test intersection of two straight lines and check parameter values."""
        # Create two crossing lines in the XY plane
        line1 = create_line_segment(
            (0, 0, 0), (2, 2, 0)
        )  # u from 0 to 1, point (1,1,0) is at u=0.5
        line2 = create_line_segment(
            (0, 2, 0), (2, 0, 0)
        )  # v from 0 to 1, point (1,1,0) is at v=0.5

        intersections = IntersectBSplines(line1, line2, tolerance=1e-6)

        assert len(intersections) == 1

        # The intersection point (1,1,0) should correspond to u=0.5 and v=0.5
        assert abs(intersections[0]["parmOnCurve1"] - 0.5) < 1e-6
        assert abs(intersections[0]["parmOnCurve2"] - 0.5) < 1e-6

    def test_intersect_spline_and_circle(self):
        radius = 10
        length = 36

        outer = create_bspline_from_points([(0.8, 1), (1.1, 0.35), (1.0, 0)])
        inner = create_bspline_from_points([(0.9, 0), (0.85, 0.35), (0.7, 1)])
        num_points = 40
        points = *[outer.Value(i / num_points) for i in range(num_points + 1)], *[
            inner.Value(i / num_points) for i in range(num_points + 1)
        ]
        points = [(p.X() * radius, p.Y() * length) for p in points]

        guide1 = create_bspline_from_points(points)

        def rotate(curve: Geom_BSplineCurve, deg: float):
            trsf = gp_Trsf()
            trsf.SetRotation(
                gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)), math.radians(deg)
            )
            curve2 = clone_bspline(curve)
            curve2.Transform(trsf)
            return curve2

        guides = [guide1, rotate(guide1, 90), rotate(guide1, 180), rotate(guide1, 270)]

        def to_circle(v: gp_Pnt):
            r = math.sqrt(v.X() * v.X() + v.Z() * v.Z())
            # print(f'v=[{v.X()}, {v.Y()}, {v.Z()}]')
            # print(f'r={r}')
            ax2 = gp_Ax2(gp_Pnt(0, v.Y(), 0), gp_Dir(0, -1, 0), gp_Dir(1, 0, 0))
            circle = Geom_Circle(ax2, r)
            bspline = GeomConvert.CurveToBSplineCurve_s(circle)
            return bspline

        profiles = [to_circle(guide1.Value(0)), to_circle(guide1.Value(1))]

        for i in range(len(profiles)):
            for j in range(len(guides)):
                intersections = IntersectBSplines(
                    profiles[i], guides[j], tolerance=1e-6
                )
                # print(f"distance({i},{j})=", profiles[i].Value(0).Distance(guides[j].Value(0)))
                # u = 0
                # p = profiles[i].Value(0)
                # print(f"profiles[{i}](0)=[{p.X()}, {p.Y()}, {p.Z()}]")
                assert len(intersections) == (
                    2 if j == 0 else 1
                ), f"#{i} profile and #{j} guide not intersect"

    def test_intersect_bspline_with_kink(self):
        """Test intersection of two curves"""
        objs = [
            {
                "poles": [
                    [0.0, 0.0, 0.0],
                    [0.95, 8.96, -1.49],
                    [3.42, -5.68, -0.76],
                    [5, -4, 1],
                    [6.9, 12.6, 10.2],
                    [5.96, -23.42, 5.70],
                    [8.0, 4.0, -3.0],
                ],
                "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "knots": [0.0, 0.35, 1.0],
                "mults": [4, 3, 4],
                "degree": 3,
                "is_periodic": False,
                "first_parameter": 0.0,
                "last_parameter": 1.0,
            },
            {
                "poles": [[5, -4, 1], [8.0, 4.0, -13.0]],
                "weights": [1.0, 1.0],
                "knots": [0.0, 1.0],
                "mults": [2, 2],
                "degree": 1,
                "is_periodic": False,
                "first_parameter": 0.0,
                "last_parameter": 1.0,
            },
        ]
        curve1, curve2 = load_bsplines_from_object(objs)

        tolerance = 1e-6
        intersections = IntersectBSplines(curve1, curve2, tolerance=tolerance)

        # Should find at least one intersection near (0.5, 0.25, 0)
        assert len(intersections) == 1

        found_intersection = False
        for intersection in intersections:
            p = intersection["point"]
            t = gp_Pnt(5, -4, 1)
            if (
                abs(p.X() - t.X()) < tolerance
                and abs(p.Y() - t.Y()) < tolerance
                and abs(p.Z() - t.Z()) < tolerance
            ):
                found_intersection = True
                break

        if not found_intersection:
            for intersection in intersections:
                p = intersection["point"]
                # print(f'p=[{p.X()}, {p.Y()}, {p.Z()}]')

        assert found_intersection

    def test_intersect_tangent_curves(self):
        """Test intersection of two curves that are tangent at a point."""
        # Create a parabola-like curve
        curve1 = create_bspline_from_points(
            [(0.2, 0, 0.2), (0.5, 0.25, 0), (1, 0.4, -0.2)]
        )
        curve2 = create_line_segment((0, 0, 0), (1, 0.5, 0))

        # save_bsplines_to_file([curve1, curve2], "curve1.json")

        tolerance = 1e-6
        intersections = IntersectBSplines(curve1, curve2, tolerance=tolerance)

        # u, v = 0.5, 0.5
        # print(f'curve1.Value({u})={curve1.Value(u).X()}, {curve1.Value(u).Y()}, {curve1.Value(u).Z()}')
        # print(f'curve2.Value({v})={curve2.Value(v).X()}, {curve2.Value(v).Y()}, {curve2.Value(v).Z()}')
        # print(f'distance={curve1.Value(u).Distance(curve2.Value(v)))}')

        # Should find at least one intersection near (0.5, 0.25, 0)
        assert len(intersections) >= 1

        found_tangent_intersection = False
        for intersection in intersections:
            p = intersection["point"]
            t = gp_Pnt(0.5, 0.25, 0)
            if (
                abs(p.X() - t.X()) < tolerance
                and abs(p.Y() - t.Y()) < tolerance
                and abs(p.Z() - t.Z()) < tolerance
            ):
                found_tangent_intersection = True
                break

        if not found_tangent_intersection:
            for intersection in intersections:
                p = intersection["point"]
                # print(f'p=[{p.X()}, {p.Y()}, {p.Z()}]')

        assert found_tangent_intersection

    def test_line_line_intersection_3d_specific_case(self):
        """Test a specific 3D line-line intersection case."""
        p1_start = gp_Pnt(1.7158016474124649, 70.0, 3.774703119306688)
        p1_end = gp_Pnt(0.4615237049031605, 69.99999999999999, 2.1221380939367966)
        p2_start = gp_Pnt(1.8186963349214254, 65.40824062829344, 1.4524829412147489)
        p2_end = gp_Pnt(0.5197234841721123, 74.5, 3.7052765158278875)

        t, s, possible_intersect = line_line_intersection_3d(
            p1_start, p1_end, p2_start, p2_end, 1.0015
        )

        # Based on the nature of the function, we expect it to return parameters
        # and a boolean indicating possible intersection.
        # The exact values of t and s would require manual calculation or a reference.
        # For now, we assert that possible_intersect is True and t, s are within reasonable bounds.
        assert possible_intersect is True
        assert 0.0 <= t <= 1.0
        assert 0.0 <= s <= 1.0

        # Optionally, calculate the actual intersection point and check distance
        # This would require more complex setup or a known expected intersection point.
        # For a basic test, checking the boolean and parameter ranges is a good start.

    def test_is_point_on_line_segment(self):
        """Test the is_point_on_line_segment function."""
        p_start = gp_Pnt(0, 0, 0)
        p_end = gp_Pnt(10, 0, 0)

        # Case 1: Point directly on the segment
        point_on_segment = gp_Pnt(5, 0, 0)
        t, is_on_segment = is_point_on_line_segment(point_on_segment, p_start, p_end)
        assert is_on_segment is True
        assert abs(t - 0.5) < 1e-7

        # Case 2: Point at the start of the segment
        point_at_start = gp_Pnt(0, 0, 0)
        t, is_on_segment = is_point_on_line_segment(point_at_start, p_start, p_end)
        assert is_on_segment is True
        assert abs(t - 0.0) < 1e-7

        # Case 3: Point at the end of the segment
        point_at_end = gp_Pnt(10, 0, 0)
        t, is_on_segment = is_point_on_line_segment(point_at_end, p_start, p_end)
        assert is_on_segment is True
        assert abs(t - 1.0) < 1e-7

        # Case 4: Point collinear but outside the segment (before start)
        point_before_segment = gp_Pnt(-1, 0, 0)
        t, is_on_segment = is_point_on_line_segment(
            point_before_segment, p_start, p_end
        )
        assert is_on_segment is False
        assert t < 0.0  # Parameter should be less than 0

        # Case 5: Point collinear but outside the segment (after end)
        point_after_segment = gp_Pnt(11, 0, 0)
        t, is_on_segment = is_point_on_line_segment(point_after_segment, p_start, p_end)
        assert is_on_segment is False
        assert t > 1.0  # Parameter should be greater than 1

        # Case 6: Point not collinear (distance too large)
        point_not_collinear = gp_Pnt(5, 1, 0)
        t, is_on_segment = is_point_on_line_segment(point_not_collinear, p_start, p_end)
        assert is_on_segment is False

        # Case 7: Point very close to the line (within tolerance)
        # The max_curvature parameter in is_point_on_line_segment affects max_distance.
        # Let's use a max_curvature that results in a small max_distance for a tight check.
        # For a straight line, max_curvature = 1.0005 gives a very small sagitta.
        # A point at (5, 1e-8, 0) should be considered on the line with default tolerance.
        point_close_to_line = gp_Pnt(5, 1e-8, 0)
        t, is_on_segment = is_point_on_line_segment(
            point_close_to_line, p_start, p_end, max_curvature=1.0000001
        )
        assert is_on_segment is True
        assert abs(t - 0.5) < 1e-7

        # Case 8: Point not on line which is also a point
        point_not_on_linear = gp_Pnt(5, 1, 0)
        t, is_on_segment = is_point_on_line_segment(
            point_not_on_linear, p_start, p_start
        )
        assert is_on_segment is False

        # Case 9: Point on line which is also a point
        point_not_on_linear = gp_Pnt(5, 0, 0)
        t, is_on_segment = is_point_on_line_segment(
            point_not_on_linear, p_start, p_start
        )
        assert is_on_segment is False


if __name__ == "__main__":
    if 0:
        pytest.main(
            [
                f"{__file__}::TestIntersectBSplines::test_intersect_spline_and_circle",
                "-v",
            ]
        )
    else:
        pytest.main([f"{__file__}", "-v"])
