import pytest
import sys
import os
import numpy as np
from OCP.gp import gp_Pnt
from OCP.TColgp import TColgp_HArray1OfPnt, TColgp_Array1OfPnt
from OCP.Geom import Geom_BSplineCurve
from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src_py.ocp_gordon.internal.points_to_bspline_interpolation import PointsToBSplineInterpolation
from src_py.ocp_gordon.internal.bspline_algorithms import BSplineAlgorithms # Assuming BSplineAlgorithms is correctly implemented

# Helper function to create TColgp_HArray1OfPnt from a list of numpy arrays or gp_Pnt
def create_occ_points(points_data: list) -> TColgp_HArray1OfPnt:
    if not points_data:
        raise ValueError("Points data cannot be empty.")
    
    occ_array = TColgp_Array1OfPnt(1, len(points_data))
    for i, p in enumerate(points_data):
        if isinstance(p, gp_Pnt):
            occ_array.SetValue(i + 1, p)
        elif isinstance(p, (list, tuple, np.ndarray)):
            if len(p) != 3:
                raise ValueError("Point data must have 3 coordinates (x, y, z).")
            occ_array.SetValue(i + 1, gp_Pnt(p[0], p[1], p[2]))
        else:
            raise TypeError("Unsupported point data type. Must be gp_Pnt or list/tuple/ndarray.")
    return TColgp_HArray1OfPnt(occ_array)

class TestPointsToBSplineInterpolation:

    def test_init_validation_degree(self):
        points_data = [[0, 0, 0], [1, 1, 0]]
        occ_points = create_occ_points(points_data)
        with pytest.raises(ValueError, match="Degree must be larger than 1"):
            PointsToBSplineInterpolation(occ_points, max_degree=0)

    def test_init_validation_too_few_points(self):
        points_data = [[0, 0, 0]]
        occ_points = create_occ_points(points_data)
        with pytest.raises(ValueError, match="Too few points"):
            PointsToBSplineInterpolation(occ_points)

    def test_init_validation_params_mismatch(self):
        points_data = [[0, 0, 0], [1, 1, 0]]
        occ_points = create_occ_points(points_data)
        with pytest.raises(ValueError, match="Number of parameters"):
            PointsToBSplineInterpolation(occ_points, parameters=[0.0])

    def test_simple_interpolation_linear(self):
        points_data = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        occ_points = create_occ_points(points_data)
        interpolator = PointsToBSplineInterpolation(occ_points, max_degree=1)
        curve = interpolator.curve()

        assert isinstance(curve, Geom_BSplineCurve)
        assert curve.Degree() == 1
        assert curve.NbPoles() == len(points_data)
        
        # Check if the curve interpolates the points
        for i, p_data in enumerate(points_data):
            param = interpolator.m_params[i]
            eval_pnt = curve.Value(param)
            assert np.isclose(eval_pnt.X(), p_data[0])
            assert np.isclose(eval_pnt.Y(), p_data[1])
            assert np.isclose(eval_pnt.Z(), p_data[2])

    def test_simple_interpolation_cubic(self):
        points_data = [[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0]]
        occ_points = create_occ_points(points_data)
        interpolator = PointsToBSplineInterpolation(occ_points, max_degree=3)
        curve = interpolator.curve()

        assert isinstance(curve, Geom_BSplineCurve)
        assert curve.Degree() == 3
        # For open curves, NbPoles should be equal to NbPoints for interpolation
        assert curve.NbPoles() == len(points_data)

        # Check if the curve interpolates the points
        for i, p_data in enumerate(points_data):
            param = interpolator.m_params[i]
            eval_pnt = curve.Value(param)
            assert np.isclose(eval_pnt.X(), p_data[0])
            assert np.isclose(eval_pnt.Y(), p_data[1])
            assert np.isclose(eval_pnt.Z(), p_data[2])

    def test_closed_curve_interpolation(self):
        # A square to test closed curve interpolation
        points_data = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]]
        occ_points = create_occ_points(points_data)
        
        # For a closed curve, the last point is a duplicate of the first for interpolation purposes
        # The actual number of unique points is len(points_data) - 1
        interpolator = PointsToBSplineInterpolation(occ_points, max_degree=3, continuous_if_closed=True)
        curve = interpolator.curve()

        assert isinstance(curve, Geom_BSplineCurve)
        assert curve.IsClosed()
        assert interpolator.is_closed()
        
        # For closed curves, the number of poles is typically NbPoints + Degree - 1 (for C2 continuity)
        # The algorithm removes the last parameter, so it's based on unique points.
        # Number of unique points = 4. Degree = 3. NbPoles = 5 + 3 - 1 = 7
        assert curve.NbPoles() == len(points_data) + interpolator.degree() - 1

        # Check if the curve interpolates the unique points
        unique_points_data = points_data[:-1] # Exclude the last duplicate point
        for i, p_data in enumerate(unique_points_data):
            param = interpolator.m_params[i]
            eval_pnt = curve.Value(param)
            assert np.isclose(eval_pnt.X(), p_data[0], atol=1e-6)
            assert np.isclose(eval_pnt.Y(), p_data[1], atol=1e-6)
            assert np.isclose(eval_pnt.Z(), p_data[2], atol=1e-6)
        
        # Check the closing point
        eval_pnt_end = curve.Value(interpolator.m_params[-1])
        assert np.isclose(eval_pnt_end.X(), points_data[0][0], atol=1e-6)
        assert np.isclose(eval_pnt_end.Y(), points_data[0][1], atol=1e-6)
        assert np.isclose(eval_pnt_end.Z(), points_data[0][2], atol=1e-6)


    def test_degree_calculation(self):
        points_data = [[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0], [4, 0, 0]]
        occ_points = create_occ_points(points_data)
        
        # Test with max_degree less than (num_points - 1)
        interpolator = PointsToBSplineInterpolation(occ_points, max_degree=2)
        assert interpolator.degree() == 2

        # Test with max_degree greater than or equal to (num_points - 1)
        interpolator = PointsToBSplineInterpolation(occ_points, max_degree=4)
        assert interpolator.degree() == 4 # num_points - 1

        # Test with closed curve
        closed_points_data = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]]
        occ_closed_points = create_occ_points(closed_points_data)
        interpolator_closed = PointsToBSplineInterpolation(occ_closed_points, max_degree=3, continuous_if_closed=True)
        # For closed curves, max_degree is (num_unique_points - 1)
        # num_unique_points = 4, so max_degree = 3
        assert interpolator_closed.degree() == 3

    def test_is_closed(self):
        # Open curve
        points_data = [[0, 0, 0], [1, 1, 0], [2, 0, 0]]
        occ_points = create_occ_points(points_data)
        interpolator = PointsToBSplineInterpolation(occ_points)
        assert not interpolator.is_closed()

        # Closed curve (first and last points are the same)
        closed_points_data = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]]
        occ_closed_points = create_occ_points(closed_points_data)
        interpolator_closed = PointsToBSplineInterpolation(occ_closed_points, continuous_if_closed=True)
        assert interpolator_closed.is_closed()

        # Closed curve, but continuous_if_closed is False
        interpolator_not_c2_closed = PointsToBSplineInterpolation(occ_closed_points, continuous_if_closed=False)
        assert not interpolator_not_c2_closed.is_closed() # Should be false if C2 continuity is not enforced

    def test_needs_shifting(self):
        # Open curve, degree 3 (odd)
        points_data = [[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0]]
        occ_points = create_occ_points(points_data)
        interpolator = PointsToBSplineInterpolation(occ_points, max_degree=3)
        assert not interpolator.is_closed() # Not closed
        assert not interpolator.needs_shifting()

        # Closed curve, degree 3 (odd)
        closed_points_data = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]]
        occ_closed_points = create_occ_points(closed_points_data)
        interpolator_closed_odd_degree = PointsToBSplineInterpolation(occ_closed_points, max_degree=3, continuous_if_closed=True)
        assert interpolator_closed_odd_degree.is_closed()
        assert interpolator_closed_odd_degree.degree() == 3 # Odd degree
        assert not interpolator_closed_odd_degree.needs_shifting()

        # Closed curve, degree 2 (even)
        # Need at least 3 unique points for degree 2
        closed_points_data_even_degree = [[0,0,0], [1,0,0], [0.5,1,0], [0,0,0]]
        occ_closed_points_even_degree = create_occ_points(closed_points_data_even_degree)
        interpolator_closed_even_degree = PointsToBSplineInterpolation(occ_closed_points_even_degree, max_degree=2, continuous_if_closed=True)
        assert interpolator_closed_even_degree.is_closed()
        assert interpolator_closed_even_degree.degree() == 2 # Even degree
        assert interpolator_closed_even_degree.needs_shifting()

    def test_call_method(self):
        points_data = [[0, 0, 0], [1, 1, 0], [2, 0, 0]]
        occ_points = create_occ_points(points_data)
        interpolator = PointsToBSplineInterpolation(occ_points, max_degree=2)
        curve = interpolator() # Using __call__
        assert isinstance(curve, Geom_BSplineCurve)
        assert curve.Degree() == 2

if __name__ == "__main__":
    # pytest.main([f'{__file__}::TestPointsToBSplineInterpolation::test_closed_curve_interpolation', "-v"])
    pytest.main([f'{__file__}', "-v"])
