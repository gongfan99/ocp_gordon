"""
Surface skinning from curves.

This module provides functionality to create surfaces by skinning
a set of compatible B-spline curves.
"""

from typing import List, Optional, Tuple

import numpy as np
from OCP.BSplCLib import BSplCLib  # For C++ BSplCLib equivalent
from OCP.Geom import (
    Geom_BSplineCurve,
    Geom_BSplineSurface,
    Geom_Curve,
    Geom_TrimmedCurve,
)
from OCP.GeomAPI import (  # Equivalent to Geom2dAPI_ProjectPointOnCurve for 3D
    GeomAPI_Interpolate,
)
from OCP.GeomConvert import GeomConvert
from OCP.gp import gp_Pnt
from OCP.Precision import Precision  # For C++ Precision equivalent
from OCP.TColgp import (
    TColgp_Array1OfPnt,
    TColgp_Array2OfPnt,
    TColgp_HArray1OfPnt,
    TColgp_HArray1OfPnt2d,
)
from OCP.TColStd import (
    TColStd_Array1OfInteger,
    TColStd_Array1OfReal,
    TColStd_Array2OfReal,
    TColStd_HArray1OfInteger,
    TColStd_HArray1OfReal,
)

from .bspline_algorithms import BSplineAlgorithms
from .error import ErrorCode, error  # Import ErrorCode
from .points_to_bspline_interpolation import (  # Import PointsToBSplineInterpolation
    PointsToBSplineInterpolation,
)


# Helper function to clamp BSpline (equivalent to C++ clampBSpline)
def clamp_bspline(curve: Geom_BSplineCurve):
    """
    Clamps a periodic B-spline curve by removing periodicity.
    Equivalent to C++ clampBSpline.
    """
    if not curve.IsPeriodic():
        return

    # C++ logic: SetNotPeriodic, trim to original range, convert back.
    curve.SetNotPeriodic()

    first_param = curve.FirstParameter()
    last_param = curve.LastParameter()

    # Create a new curve by trimming to the original parameter range
    trimmed_curve = Geom_TrimmedCurve(curve, first_param, last_param)

    # Convert back to BSplineCurve
    new_curve = GeomConvert.CurveToBSplineCurve_s(trimmed_curve)

    # Update the original curve handle (or return new_curve if preferred)
    # For in-place modification, we need to be careful with handles.
    # Let's assume we can modify the curve object directly or replace it.
    # In Python, modifying the object passed might not update the original if it's a copy.
    # However, Geom_BSplineCurve is likely a handle-based object in OCP.
    # For safety, let's return the new curve and let the caller assign it.
    # If the caller expects in-place modification, this needs adjustment.
    # For now, let's assume the caller will handle assignment.
    # If the C++ `curve = GeomConvert::CurveToBSplineCurve(c);` implies assignment,
    # then we should return the new curve.
    return new_curve


class CurvesToSurface:
    """
    Creates a surface by skinning a set of compatible B-spline curves.

    The curves must have compatible parameterizations, degrees, and
    knot vectors for successful skinning. This implementation aims to
    mirror the C++ version's functionality.
    """

    def __init__(
        self,
        curves: list[Geom_Curve],
        parameters: list[float] | None = None,
        continuous_if_closed: bool = False,
        tolerance: float = 1e-14,
    ):
        """
        Initialize the surface skinner.

        Args:
            curves: List of curves to be interpolated. These will be converted to B-splines.
            parameters: Optional list of parameters for v-direction interpolation. If None, they will be calculated.
            continuous_if_closed: If True, attempts to make a C2 continuous surface at the start/end junction
                                  if the first and last curve are the same.
            tolerance: Construction tolerance.
        """
        self._input_curves_raw = curves  # Store raw curves for potential conversion
        self._parameters = parameters if parameters is not None else []
        self._continuous_if_closed = continuous_if_closed
        self._tolerance = tolerance
        self._max_degree = (
            3  # Default max degree, matching C++ default for interpolation
        )

        self._has_performed = False
        self._skinned_surface = None

        self._input_curves: list[Geom_BSplineCurve] = []
        self._compatible_splines: list[Geom_BSplineCurve] = (
            []
        )  # Initialize _compatible_splines here

        # Convert all curves to bspline curves and store them
        for curve in self._input_curves_raw:
            self._input_curves.append(GeomConvert.CurveToBSplineCurve_s(curve))

        BSplineAlgorithms.match_degree(self._input_curves)

        # Calculate parameters if not provided
        if not self._parameters:
            self._calculate_parameters()

        # Ensure compatible splines for later use, if not already done by _calculate_parameters
        if not self._compatible_splines:  # Check if it's still empty
            self._compatible_splines = (
                BSplineAlgorithms.create_common_knots_vector_curve(
                    self._input_curves, self._tolerance
                )
            )

    def set_max_degree(self, degree: int):
        """
        Sets the maximum interpolation degree of the splines in skinning direction (v-direction).

        Args:
            degree: Maximum degree for v-direction interpolation.
        """
        if degree <= 0:
            raise ValueError("Degree must be positive.")
        self._max_degree = degree
        self.invalidate()  # Invalidate cached surface

    def get_parameters(self) -> list[float]:
        """Returns the parameters at the profile curves (v-direction)."""
        return self._parameters

    def surface(self) -> Geom_BSplineSurface:
        """Returns the skinned surface."""
        if not self._has_performed:
            self.perform()
        if self._skinned_surface is None:
            raise RuntimeError("Surface could not be created.")
        return self._skinned_surface

    def invalidate(self):
        """Invalidates the cached surface, forcing recomputation on next call."""
        self._has_performed = False
        self._skinned_surface = None

    def _calculate_parameters(self):
        """
        Calculates parameters for the v-direction based on control points.
        Mirrors C++ CurvesToSurface::CalculateParameters.
        """
        if not self._input_curves:
            return

        # Ensure common knot vector is created if not already
        if not self._compatible_splines:
            self._compatible_splines = (
                BSplineAlgorithms.create_common_knots_vector_curve(
                    self._input_curves, self._tolerance
                )
            )

        # Create a matrix of control points of all B-splines
        # (splines do have the same amount of control points now after matchDegree)
        first_curve = self._compatible_splines[0]
        num_poles_u = first_curve.NbPoles()
        num_splines = len(self._compatible_splines)

        # TColgp_Array2OfPnt(RowMin, RowMax, ColMin, ColMax)
        # Rows correspond to poles in U direction, Columns correspond to splines (V direction)
        control_points_matrix = TColgp_Array2OfPnt(1, num_poles_u, 1, num_splines)

        for spline_idx, spline in enumerate(self._compatible_splines, 1):
            for pole_idx in range(1, num_poles_u + 1):
                control_points_matrix.SetValue(
                    pole_idx, spline_idx, spline.Pole(pole_idx)
                )

        # Compute parameters using the surface-specific algorithm
        # C++ uses BSplineAlgorithms::computeParamsBSplineSurf(controlPoints)
        # Now that compute_params_bspline_surf is implemented in bspline_algorithms.py, use it.

        # The C++ version uses a default alpha of 0.5 for centripetal parameterization.
        # The `compute_params_bspline_surf` returns (u_params, v_params).
        # In CurvesToSurface, `_parameters` refers to the parameters in the v-direction.
        _, v_params = BSplineAlgorithms.compute_params_bspline_surf(
            control_points_matrix, alpha=0.5
        )
        self._parameters = v_params

    def perform(self):
        """
        Build the surface by skinning the curves.
        Mirrors C++ CurvesToSurface::Perform.
        """
        # check amount of given parameters
        if len(self._input_curves) < 2:
            return

        if len(self._parameters) != len(self._input_curves):
            raise error(
                "The amount of given parameters has to be equal to the amount of given B-splines!",
                ErrorCode.MATH_ERROR,
            )

        # check if all curves are closed
        # C++ uses BSplineAlgorithms::scale(_inputCurves) * BSplineAlgorithms::REL_TOL_CLOSED
        tolerance = (
            BSplineAlgorithms.scale(self._input_curves)
            * BSplineAlgorithms.REL_TOL_CLOSED
        )

        # Check if first and last input curves are equal
        make_closed = (
            self._continuous_if_closed
            and len(self._input_curves) > 0
            and self._input_curves[0].IsEqual(self._input_curves[-1], tolerance)
        )

        n_curves = len(self._input_curves)

        # create a common knot vector for all splines if not already done
        if not self._compatible_splines:
            self._compatible_splines = (
                BSplineAlgorithms.create_common_knots_vector_curve(
                    self._input_curves, self._tolerance
                )
            )

        first_curve = self._compatible_splines[0]
        num_control_points_u = first_curve.NbPoles()

        degree_v = 0
        degree_u = first_curve.Degree()
        knots_v = None
        mults_v = None

        # C++ uses Handle(TColgp_HArray2OfPnt) cpSurf;
        # Python equivalent: a list of lists or a numpy array, then convert to TColgp_Array2OfPnt
        # Or directly use TColgp_HArray1OfPnt for interpolation points.
        cp_surf = TColgp_Array2OfPnt()
        interp_spline = None

        # C++ uses Handle(TColgp_HArray1OfPnt) interpPointsVDir = new TColgp_HArray1OfPnt(1, static_cast<Standard_Integer>(nCurves));
        # This array will hold the poles for a given U index across all V curves.
        interp_points_v_dir = TColgp_HArray1OfPnt(1, n_curves)

        # C++ iterates from cpUIdx = 1 to numControlPointsU
        for cp_u_idx in range(1, num_control_points_u + 1):
            # Populate interpPointsVDir with poles from each compatible spline at the current U index
            for cp_v_idx in range(1, n_curves + 1):
                pole = self._compatible_splines[cp_v_idx - 1].Pole(cp_u_idx)
                interp_points_v_dir.SetValue(cp_v_idx, pole)

            # C++ uses PointsToBSplineInterpolation interpol(interpPointsVDir, _parameters, _maxDegree, makeClosed);
            # Python equivalent: PointsToBSplineInterpolation(points, parameters, max_degree, continuous_if_closed)
            interpolator = PointsToBSplineInterpolation(
                points=interp_points_v_dir,
                parameters=self._parameters,
                max_degree=self._max_degree,
                continuous_if_closed=make_closed,
            )

            # Call curve() to get the interpolated spline
            interp_spline = interpolator.curve()

            if interp_spline is None:
                raise RuntimeError(f"Interpolation failed for U index {cp_u_idx}")

            if make_closed:
                # Apply clamping if the curve is closed and we need continuity
                clamped_spline = clamp_bspline(interp_spline)
                if (
                    clamped_spline
                ):  # clamp_bspline returns None if not periodic or if modification failed
                    interp_spline = clamped_spline

            if cp_u_idx == 1:
                degree_v = interp_spline.Degree()

                # Get knots and multiplicities for V direction
                knots_v_list = []
                for i in range(1, interp_spline.NbKnots() + 1):
                    knots_v_list.append(interp_spline.Knot(i))
                knots_v = TColStd_Array1OfReal(1, len(knots_v_list))
                for i, k in enumerate(knots_v_list, 1):
                    knots_v.SetValue(i, k)

                mults_v_list = []
                for i in range(1, interp_spline.NbKnots() + 1):
                    mults_v_list.append(interp_spline.Multiplicity(i))
                mults_v = TColStd_Array1OfInteger(1, len(mults_v_list))
                for i, m in enumerate(mults_v_list, 1):
                    mults_v.SetValue(i, m)

                # Initialize the surface control points array
                # C++ uses Handle(TColgp_HArray2OfPnt) cpSurf;
                # cpSurf = new TColgp_HArray2OfPnt(1, static_cast<Standard_Integer>(numControlPointsU), 1, interpSpline->NbPoles());
                # Python equivalent: TColgp_Array2OfPnt(RowMin, RowMax, ColMin, ColMax)
                # Rows = U direction (num_control_points_u), Cols = V direction (interpSpline.NbPoles())
                cp_surf = TColgp_Array2OfPnt(
                    1, num_control_points_u, 1, interp_spline.NbPoles()
                )

            # the final surface control points are the control points resulting from the interpolation
            for i in range(1, interp_spline.NbPoles() + 1):
                cp_surf.SetValue(cp_u_idx, i, interp_spline.Pole(i))

            # check degree always the same
            # C++ asserts degreeV == interpSpline->Degree()
            if cp_u_idx > 1 and degree_v != interp_spline.Degree():
                raise error(
                    f"Inconsistent degree_v found at U index {cp_u_idx}. Expected {degree_v}, got {interp_spline.Degree()}.",
                    ErrorCode.MATH_ERROR,
                )

        # Get U knots and multiplicities from the first compatible spline
        knots_u_list = []
        for i in range(1, first_curve.NbKnots() + 1):
            knots_u_list.append(first_curve.Knot(i))
        knots_u = TColStd_Array1OfReal(1, len(knots_u_list))
        for i, k in enumerate(knots_u_list, 1):
            knots_u.SetValue(i, k)

        mults_u_list = []
        for i in range(1, first_curve.NbKnots() + 1):
            mults_u_list.append(first_curve.Multiplicity(i))
        mults_u = TColStd_Array1OfInteger(1, len(mults_u_list))
        for i, m in enumerate(mults_u_list, 1):
            mults_u.SetValue(i, m)

        # Construct the final Geom_BSplineSurface

        # Check if knots_v and mults_v were successfully populated
        if knots_v is None or mults_v is None:
            raise RuntimeError("Failed to obtain V direction knots and multiplicities.")

        # Number of poles in U direction: num_control_points_u
        # Number of poles in V direction: interp_spline.NbPoles()

        self._skinned_surface = Geom_BSplineSurface(
            cp_surf,
            knots_u,
            knots_v,
            mults_u,
            mults_v,
            degree_u,
            degree_v,
        )

        self._has_performed = True
