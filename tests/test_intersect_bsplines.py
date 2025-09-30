"""
Test cases for intersect_bsplines module.

This module tests the B-spline curve intersection functionality
using recursive subdivision and optimization methods.
"""

import pytest
import numpy as np
import sys
import os
from OCP.Geom import Geom_BSplineCurve
from OCP.GeomAPI import GeomAPI_PointsToBSpline
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.gp import gp_Pnt
from OCP.GeomAbs import GeomAbs_Shape
import math

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the module to test
from src_py.ocp_gordon.internal.intersect_bsplines import IntersectBSplines, BoundingBox


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
        p2 = line2.Value(v) # Re-evaluate p2 using the parameter from the intersection result
        
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
        p2 = line2.Value(v) # Re-evaluate p2 using the parameter from the intersection result
        
        assert abs(p1.X() - 1.0) < 1e-6
        assert abs(p1.Y() - 1.0) < 1e-6
        assert abs(p1.Z() - 0.0) < 1e-6
        assert abs(p2.X() - 1.0) < 1e-6
        assert abs(p2.Y() - 1.0) < 1e-6
        assert abs(p2.Z() - 0.0) < 1e-6
    
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
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
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
        ]
        correct_intersec_points = [(0, 0, 0), (0, 0, 0), (1, 1, 1), (0.5, 0.25, 0),]
        for i in range(len(correct_intersec_points)):
            curve1 = create_bspline_from_points(dataset[2*i])
            curve2 = create_bspline_from_points(dataset[2*i+1])
       
            intersections = IntersectBSplines(curve1, curve2, tolerance=1e-6)
            
            # Should find at least one intersection near (1, 1, 1)
            assert len(intersections) >= 1, f'#{i} dataset failed'
            
            # Check that at least one intersection is near (1, 1, 1)
            found_intersection = False
            for intersection in intersections:
                p = intersection["point"]
                t = gp_Pnt(*correct_intersec_points[i])
                # print(f'distance={p.Distance(t)}')
                if (abs(p.X() - t.X()) < 1e-6 and 
                    abs(p.Y() - t.Y()) < 1e-6 and 
                    abs(p.Z() - t.Z()) < 1e-6):
                    found_intersection = True
                    break
            
            assert found_intersection, f'#{i} dataset failed'
    
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
        line1 = create_line_segment((0, 0, 0), (2, 2, 0)) # u from 0 to 1, point (1,1,0) is at u=0.5
        line2 = create_line_segment((0, 2, 0), (2, 0, 0)) # v from 0 to 1, point (1,1,0) is at v=0.5
        
        intersections = IntersectBSplines(line1, line2, tolerance=1e-6)
        
        assert len(intersections) == 1
        
        # The intersection point (1,1,0) should correspond to u=0.5 and v=0.5
        assert abs(intersections[0]["parmOnCurve1"] - 0.5) < 1e-6
        assert abs(intersections[0]["parmOnCurve2"] - 0.5) < 1e-6

    def test_intersect_tangent_curves(self):
        """Test intersection of two curves that are tangent at a point."""
        # Create a parabola-like curve
        curve1 = create_bspline_from_points([(0.2, 0, 0), (0.5, 0.25, 0), (1, 0.4, 0)])
        curve2 = create_line_segment((0, 0, 0), (1, 0.5, 0))

        intersections = IntersectBSplines(curve1, curve2, tolerance=1e-7)
        
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
            if (abs(p.X() - t.X()) < 1e-6 and 
                abs(p.Y() - t.Y()) < 1e-6 and 
                abs(p.Z() - t.Z()) < 1e-6):
                found_tangent_intersection = True
                break
        
        if not found_tangent_intersection:
            for intersection in intersections:
                p = intersection["point"]
                print(f'p=[{p.X()}, {p.Y()}, {p.Z()}]')

        assert found_tangent_intersection


if __name__ == "__main__":
    if 0:
        pytest.main([f'{__file__}::TestIntersectBSplines::test_intersect_3d_curves', "-v"])
    else:
        pytest.main([f'{__file__}', "-v"])
