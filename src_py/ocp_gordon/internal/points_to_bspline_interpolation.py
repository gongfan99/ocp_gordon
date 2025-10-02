"""
Point interpolation to B-spline curves.

This module provides functionality to create B-spline curves
that interpolate a set of points using the Park (2000) algorithm
for node and knot selection.
"""

import numpy as np
from OCP.Geom import Geom_BSplineCurve, Geom_TrimmedCurve
from OCP.GeomConvert import GeomConvert
from OCP.BSplCLib import BSplCLib
from OCP.TColgp import TColgp_Array1OfPnt, TColgp_HArray1OfPnt
from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger, TColStd_HArray1OfReal
from OCP.gp import gp_Pnt
from typing import List, Tuple, Optional, Union

from .bspline_algorithms import BSplineAlgorithms


class PointsToBSplineInterpolation:
    """
    Creates a B-spline curve that interpolates a set of points.
    
    Implements the b-spline interpolation algorithm as described by
    Park (2000): Choosing nodes and knots in closed B-spline curve interpolation to point data
    """
    
    def __init__(self,
                 points: TColgp_HArray1OfPnt,
                 parameters: list[float] | None = None,
                 max_degree: int = 3,
                 continuous_if_closed: bool = False):
        """
        Initialize the point interpolator.
        
        Args:
            points: Points to interpolate
            parameters: Custom parameter values (optional)
            max_degree: Maximum degree of the B-spline
            continuous_if_closed: Whether to ensure C2 continuity for closed curves
        """
        self.m_pnts = points
        self.m_degree = max_degree
        self.m_C2Continuous = continuous_if_closed
        
        # Use provided parameters or compute them
        if parameters is not None:
            self.m_params = parameters
        else:
            self.m_params = BSplineAlgorithms.compute_params_bspline_curve(points)
        
        # Validate inputs
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate input parameters."""
        if self.m_degree < 1:
            raise ValueError("Degree must be larger than 1 in PointsToBSplineInterpolation!")
        
        if self.m_pnts is None:
            raise ValueError("No points given in PointsToBSplineInterpolation")
        
        if self.m_pnts.Length() < 2:
            raise ValueError("Too few points in PointsToBSplineInterpolation")
        
        if len(self.m_params) != self.m_pnts.Length():
            raise ValueError(f"Number of parameters {len(self.m_params)} and points {self.m_pnts.Length()} don't match in PointsToBSplineInterpolation")
    
    def curve(self) -> Geom_BSplineCurve:
        """Returns the interpolation curve using Park (2000) algorithm."""
        degree = self.degree()
        
        # Work with a copy of parameters
        params = self.m_params.copy()
        
        # Generate knots using Park (2000) algorithm
        knots = BSplineAlgorithms.knots_from_curve_parameters(
            params, degree, self.is_closed()
        )
        
        if self.is_closed():
            # Remove the last parameter for closed curves (implicitly included by wrapping)
            params = params[:-1]
        
        # Convert to OpenCASCADE arrays
        occ_knots = BSplineAlgorithms.to_array(knots)
        occ_params = BSplineAlgorithms.to_array(params)
        
        # Compute B-spline basis matrix
        # bspl_mat is a numpy array (n_params x n_params + degree)
        bspl_mat = BSplineAlgorithms.bspline_basis_mat(
            degree, occ_knots.Array1(), occ_params.Array1()
        )
        
        # Build left hand side of the linear system (A * C = P)
        # A is the basis matrix, C are control points, P are interpolation points
        n_params = len(params)
        lhs = np.zeros((n_params, n_params))
        
        # Fill the matrix with basis function values
        for i in range(n_params):
            for j in range(n_params):
                lhs[i, j] = bspl_mat[i, j]
        
        if self.is_closed():
            # Set continuity constraints for closed curves by wrapping control points
            # The first 'degree' control points are influenced by the last 'degree' basis functions
            # and vice-versa. This effectively wraps the control points.
            for j in range(degree):
                for i in range(n_params):
                    lhs[i, j] += bspl_mat[i, n_params + j]
        
        # Right hand side - point coordinates
        # Store points in a numpy array for easier manipulation
        rhs_pnts = np.zeros((n_params, 3)) # (x, y, z)
        
        for i in range(n_params):
            p = self.m_pnts(i + 1) # OCP arrays are 1-indexed
            rhs_pnts[i, 0] = p.X()
            rhs_pnts[i, 1] = p.Y()
            rhs_pnts[i, 2] = p.Z()
        
        # Solve linear system using numpy.linalg.solve
        try:
            # cp_coords will be (n_params, 3) where each row is (cx, cy, cz)
            cp_coords = np.linalg.solve(lhs, rhs_pnts)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Singular Matrix in linear system solution: {e}")
        
        # Create control points
        if self.is_closed():
            n_ctrl_pnts = len(self.m_params) + degree - 1
        else:
            n_ctrl_pnts = n_params
        
        if self.needs_shifting():
            n_ctrl_pnts += 1
        
        poles = TColgp_Array1OfPnt(1, n_ctrl_pnts)
        
        # Fill control points from solution
        for i in range(n_params):
            pnt = gp_Pnt(cp_coords[i, 0], cp_coords[i, 1], cp_coords[i, 2])
            poles.SetValue(i + 1, pnt)
        
        if self.is_closed():
            # Wrap control points for closed curves
            # The first 'degree' control points are duplicates of the last 'degree'
            # from the solution, to ensure C2 continuity.
            for i in range(degree): # Only degree points are wrapped
                pnt = gp_Pnt(cp_coords[i, 0], cp_coords[i, 1], cp_coords[i, 2])
                poles.SetValue(n_params + i + 1, pnt)
        
        if self.needs_shifting():
            # Add extra control point and knot for even-degree closed curves
            deg = degree
            knots.append(knots[-1] + knots[2 * deg + 1] - knots[2 * deg])
            poles.SetValue(n_params + degree + 1, poles(degree + 1))
            
            # Shift knots back
            for i in range(len(knots)):
                knots[i] -= params[0]
        
        # Extract unique knots and their multiplicities from the flat knot vector

        occ_knots = BSplineAlgorithms.to_array(knots)
        knotsLen = BSplCLib.KnotsLength_s(occ_knots)

        occ_unique_knots = TColStd_Array1OfReal(1, knotsLen)
        occ_multiplicities = TColStd_Array1OfInteger(1, knotsLen)
        BSplCLib.Knots_s(occ_knots, occ_unique_knots, occ_multiplicities)

        # print(f'params={params}')
        # print(f'knots={knots}')
        # print(f'n_ctrl_pnts={n_ctrl_pnts}, unique_knots={occ_unique_knots}, multiplicities={occ_multiplicities}, degree={degree}')
        
        # Create final B-spline curve
        result = Geom_BSplineCurve(
            poles, occ_unique_knots, occ_multiplicities, degree, False
        )
        
        # clamp bspline
        if self.is_closed():
            result = PointsToBSplineInterpolation.clamp(result, self.m_params[0], self.m_params[-1])
        
        return result
    
    def degree(self) -> int:
        """Return the degree of the B-spline interpolation."""
        max_degree = self.m_pnts.Length() - 1
        if self.is_closed():
            max_degree -= 1
        degree = min(max_degree, self.m_degree)
        if degree <= 0:
            raise ValueError("Invalid degree computed")
        return degree
    
    def is_closed(self) -> bool:
        """Check if the curve is closed."""
        return BSplineAlgorithms.is_closed(self.m_pnts, self.m_C2Continuous)
    
    def needs_shifting(self) -> bool:
        """Check if knot shifting is needed for closed curves."""
        return (self.degree() % 2) == 0 and self.is_closed()
    
    def __call__(self) -> Geom_BSplineCurve:
        """Allow calling the object directly to get the curve."""
        return self.curve()

    @staticmethod
    def clamp(curve: Geom_BSplineCurve, first_param: float, last_param: float, tol: float = 1e-7) -> Geom_BSplineCurve:
        c = Geom_TrimmedCurve(curve, first_param, last_param)
        curve = GeomConvert.CurveToBSplineCurve_s(c)
        return curve
