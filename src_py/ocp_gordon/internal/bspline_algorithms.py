"""
B-spline algorithms and utilities.

This module provides static methods for B-spline manipulation, including
parameter computation, degree matching, knot vector manipulation, and
other B-spline operations.
"""

import bisect  # Import bisect for efficient insertion
import enum  # Import enum module
import math
from typing import List, Tuple, Union

import numpy as np
from OCP.Geom import Geom_BSplineCurve, Geom_BSplineSurface, Geom_Curve
from OCP.Geom2dAPI import Geom2dAPI_Interpolate, Geom2dAPI_ProjectPointOnCurve
from OCP.GeomAbs import GeomAbs_C2
from OCP.GeomConvert import GeomConvert
from OCP.gp import gp_Pnt, gp_Pnt2d
from OCP.TColgp import (
    TColgp_Array1OfPnt,
    TColgp_Array2OfPnt,
    TColgp_HArray1OfPnt,
    TColgp_HArray1OfPnt2d,
)
from OCP.TColStd import (
    TColStd_Array1OfInteger,
    TColStd_Array1OfReal,
    TColStd_HArray1OfInteger,
    TColStd_HArray1OfReal,
)

from .approx_result import ApproxResult
from .error import ErrorCode, error  # Import ErrorCode
from .intersect_bsplines import IntersectBSplines
from .misc import clone_bspline, clone_bspline_surface, save_bsplines_to_object


# Define SurfaceDirection enum
class SurfaceDirection(enum.Enum):
    u = 0
    v = 1
    both = 2


class BSplineAlgorithms:
    """
    Utility class with static methods for B-spline manipulation.
    """

    # Tolerance for closed curve detection
    REL_TOL_CLOSED = 1e-8  # Matching C++

    # Tolerance for comparing curve parameters
    PAR_CHECK_TOL = 1e-5  # Matching C++

    @staticmethod
    def _is_inside_tolerance(value: float, target: float, tolerance: float) -> bool:
        """Helper function to check if a value is within tolerance of a target."""
        return abs(value - target) < tolerance

    @staticmethod
    def linspace_with_breaks(
        umin: float, umax: float, n_values: int, breaks: list[float]
    ) -> list[float]:
        """
        Generates a sequence of evenly spaced values over a specified interval,
        including specified break points. This is a Python port of the C++
        LinspaceWithBreaks function.

        Args:
            umin: The starting value of the sequence.
            umax: The ending value of the sequence.
            n_values: The number of evenly spaced values to generate.
            breaks: A list of break points to include in the sequence.

        Returns:
            A list of values including the evenly spaced points and break points.
        """
        if n_values < 2:
            return [umin] if n_values == 1 else []

        du = (umax - umin) / (n_values - 1)
        result = [umin + i * du for i in range(n_values)]

        eps = 0.3

        for breakpoint in breaks:
            # Check if a point is already very close to the breakpoint (within du*eps)
            found_pos = -1
            for i, val in enumerate(result):
                if BSplineAlgorithms._is_inside_tolerance(val, breakpoint, du * eps):
                    result[i] = breakpoint  # Replace with the exact breakpoint
                    found_pos = i
                    break

            if found_pos == -1:  # If no point was found very close
                # Find closest element to decide where to insert.
                # C++ uses IsInsideTolerance(breakpoint, (0.5 + 1e-8)*du) to find a "close enough" point.
                # We need to find the closest element.

                closest_idx = -1
                min_dist = float("inf")
                for i, val in enumerate(result):
                    dist = abs(val - breakpoint)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i

                if closest_idx != -1:
                    # The C++ logic inserts before if *pos > breakpoint, and after if *pos <= breakpoint.
                    # This means if the closest element is greater than the breakpoint, insert before it.
                    # Otherwise, insert after it.
                    if result[closest_idx] > breakpoint:
                        result.insert(closest_idx, breakpoint)
                    else:
                        result.insert(closest_idx + 1, breakpoint)
                else:
                    # This case should ideally not be reached if result is not empty.
                    # If result is empty, just append.
                    if not result:
                        result.append(breakpoint)
                    else:
                        # Fallback: use bisect_left if no specific closest logic applies
                        insert_pos = bisect.bisect_left(result, breakpoint)
                        result.insert(insert_pos, breakpoint)

        return result

    @staticmethod
    def is_u_dir_closed(points: TColgp_Array2OfPnt, tolerance: float) -> bool:
        u_dir_closed = True
        ulo = points.LowerRow()
        uhi = points.UpperRow()
        # check that first row and last row are the same
        for v_idx in range(points.LowerCol(), points.UpperCol() + 1):
            pfirst = points(ulo, v_idx)  # Use __call__ for TColgp_Array2OfPnt
            pLast = points(uhi, v_idx)  # Use __call__ for TColgp_Array2OfPnt
            if not pfirst.IsEqual(pLast, tolerance):
                u_dir_closed = False
                break
        return u_dir_closed

    @staticmethod
    def is_v_dir_closed(points: TColgp_Array2OfPnt, tolerance: float) -> bool:
        v_dir_closed = True
        vlo = points.LowerCol()
        vhi = points.UpperCol()
        for u_idx in range(points.LowerRow(), points.UpperRow() + 1):
            if not points(u_idx, vlo).IsEqual(
                points(u_idx, vhi), tolerance
            ):  # Use __call__ for TColgp_Array2OfPnt
                v_dir_closed = False
                break
        return v_dir_closed

    @staticmethod
    def get_kink_parameters(curve: Geom_BSplineCurve) -> list[float]:
        if curve is None:
            raise error("Null Pointer curve", ErrorCode.NULL_POINTER)  # Use ErrorCode

        eps = 1e-8
        kinks = []
        for knot_index in range(2, curve.NbKnots()):
            if curve.Multiplicity(knot_index) == curve.Degree():
                knot = curve.Knot(knot_index)

                # Get tangent vectors at knot +/- eps
                tangent1 = curve.DN(knot + eps, 1)
                tangent2 = curve.DN(knot - eps, 1)

                # Calculate angle between tangent vectors (0 to pi radians)
                angle = tangent1.Angle(tangent2)

                # Define a tolerance for angle comparison
                angle_tolerance = 1e-4  # radians

                # A kink implies a significant change in direction, so the angle
                # between the tangent vectors should not be close to 0 or pi.
                if not (
                    abs(angle) < angle_tolerance
                    or abs(angle - math.pi) < angle_tolerance
                ):
                    kinks.append(knot)
        return kinks

    class SurfaceKinks:  # Define SurfaceKinks as a nested class
        def __init__(self):
            self.u: list[float] = []
            self.v: list[float] = []

    @staticmethod
    def get_kink_parameters_surface(
        surface: Geom_BSplineSurface,
    ) -> "BSplineAlgorithms.SurfaceKinks":
        if surface is None:
            raise error("Null Pointer surface", ErrorCode.NULL_POINTER)  # Use ErrorCode

        kinks = BSplineAlgorithms.SurfaceKinks()  # Instantiate nested class

        for knot_index in range(2, surface.NbUKnots()):
            if surface.UMultiplicity(knot_index) == surface.UDegree():
                knot = surface.UKnot(knot_index)
                kinks.u.append(knot)

        for knot_index in range(2, surface.NbVKnots()):
            if surface.VMultiplicity(knot_index) == surface.VDegree():
                knot = surface.VKnot(knot_index)
                kinks.v.append(knot)
        return kinks

    @staticmethod
    def trim_curve(
        curve: Geom_BSplineCurve, umin: float, umax: float
    ) -> Geom_BSplineCurve:
        copy = clone_bspline(curve)  # Using clone_bspline from misc.py
        copy.Segment(umin, umax)
        return copy

    @staticmethod
    def compute_params_bspline_curve(
        points: TColgp_HArray1OfPnt, alpha: float = 0.5
    ) -> list[float]:
        """
        Computes parameters of a B-spline curve at given points.

        Args:
            points: Points where parameters are computed
            alpha: Exponent for parameter computation (0.5 = centripetal method)

        Returns:
            List of computed parameters
        """
        n_points = points.Length()
        if n_points < 2:
            return [0.0, 1.0] if n_points == 1 else []

        # Compute chord lengths
        chord_lengths = []
        total_length = 0.0

        # Convert handle array to regular array for access
        # TColgp_HArray1OfPnt is a handle, need to access underlying array
        for i in range(1, n_points):
            p1 = points(i)
            p2 = points(i + 1)
            dx = p2.X() - p1.X()
            dy = p2.Y() - p1.Y()
            dz = p2.Z() - p1.Z()
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            chord_lengths.append(length**alpha)
            total_length += chord_lengths[-1]

        # Compute parameters
        params = [0.0]
        current_length = 0.0

        for length in chord_lengths:
            current_length += length
            params.append(current_length / total_length)

        return params

    @staticmethod
    def match_parameter_range(
        bsplines: list[Geom_BSplineCurve], tolerance: float = 1e-15
    ) -> None:
        """
        Matches parameter range of all B-splines to the first B-spline.

        Args:
            bsplines: List of B-splines to match (modified in place)
            tolerance: Tolerance for parameter comparison
        """
        if not bsplines:
            return

        first_spline = bsplines[0]
        umin_ref = first_spline.FirstParameter()
        umax_ref = first_spline.LastParameter()

        for spline in bsplines[1:]:
            umin = spline.FirstParameter()
            umax = spline.LastParameter()

            if abs(umin - umin_ref) > tolerance or abs(umax - umax_ref) > tolerance:
                BSplineAlgorithms.reparametrize_bspline(
                    spline, umin_ref, umax_ref, tolerance
                )

    @staticmethod
    def match_degree(bsplines: list[Geom_BSplineCurve]) -> None:
        """
        Matches degree of all B-splines by raising to maximum degree.

        Args:
            bsplines: List of B-splines to match (modified in place)
        """
        if not bsplines:
            return

        # Find maximum degree
        max_degree = max(spline.Degree() for spline in bsplines)

        # Raise degree of all splines to maximum
        for spline in bsplines:
            current_degree = spline.Degree()
            if current_degree < max_degree:
                spline.IncreaseDegree(max_degree)

    @staticmethod
    def create_common_knots_vector_curve(
        splines_vector: list[Geom_BSplineCurve], tol: float
    ) -> list[Geom_BSplineCurve]:
        """
        Creates common knots vector for given B-splines.

        Args:
            splines_vector: Vector of B-splines with different knot vectors
            tol: Tolerance for knot comparison

        Returns:
            List of B-splines with common knot vector
        """
        if not splines_vector:
            return []

        # Collect all unique knots from all splines
        all_knots = set()
        for spline in splines_vector:
            knots = BSplineAlgorithms._get_knots(spline)
            all_knots.update(knots)

        # Sort knots and determine multiplicities
        sorted_knots = sorted(all_knots)
        common_knots = []
        multiplicities = []

        for knot in sorted_knots:
            # Find maximum multiplicity across all splines for this knot
            max_mult = 0
            for spline in splines_vector:
                mult = BSplineAlgorithms._get_knot_multiplicity(spline, knot, tol)
                max_mult = max(max_mult, mult)

            common_knots.append(knot)
            multiplicities.append(max_mult)

        # Create new splines with common knot vector
        result: list[Geom_BSplineCurve] = []
        for spline in splines_vector:
            new_spline = BSplineAlgorithms._insert_knots(
                spline, common_knots, multiplicities, tol
            )
            result.append(new_spline)

        return result

    @staticmethod
    def reparametrize_bspline(
        spline: Geom_BSplineCurve, umin: float, umax: float, tol: float = 1e-15
    ) -> None:
        """
        Changes parameter range of the B-spline curve.

        Args:
            spline: B-spline to reparametrize (modified in place)
            umin: New minimum parameter
            umax: New maximum parameter
            tol: Tolerance for parameter comparison
        """
        current_umin = spline.FirstParameter()
        current_umax = spline.LastParameter()

        if abs(current_umin - umin) < tol and abs(current_umax - umax) < tol:
            return

        # Linear transformation: u_new = a * u_old + b
        a = (umax - umin) / (current_umax - current_umin)
        b = umin - a * current_umin

        # Get current knots
        knots = BSplineAlgorithms._get_knots(spline)

        # Transform knots
        new_knots = [a * knot + b for knot in knots]

        # Create TColStd_Array1OfReal with transformed knots
        n_knots = len(new_knots)
        knots_array = TColStd_Array1OfReal(1, n_knots)
        for i, knot in enumerate(new_knots, 1):
            knots_array.SetValue(i, knot)

        # Apply new knots to the spline
        spline.SetKnots(knots_array)

    @staticmethod
    def to_bsplines(curves: list[Geom_Curve]) -> list[Geom_BSplineCurve]:
        """
        Converts a curve array into a B-spline array.

        Args:
            curves: List of generic curves

        Returns:
            List of B-spline curves
        """
        result = []
        for curve in curves:
            if isinstance(curve, Geom_BSplineCurve):
                result.append(clone_bspline(curve))
            else:
                # Convert generic curve to B-spline
                # This would use OCP conversion methods
                bspline = BSplineAlgorithms._convert_to_bspline(curve)
                result.append(bspline)
        return result

    @staticmethod
    def intersections(
        spline1: Geom_BSplineCurve, spline2: Geom_BSplineCurve, tolerance: float = 3e-4
    ) -> list[tuple[float, float]]:
        """
        Returns all intersections of two B-splines.

        Args:
            spline1: First B-spline
            spline2: Second B-spline
            tolerance: Relative tolerance for intersection checking

        Returns:
            List of (parameter on spline1, parameter on spline2) pairs
        """
        # Compute average scale of the two splines
        scale1 = BSplineAlgorithms.scale(spline1)
        scale2 = BSplineAlgorithms.scale(spline2)
        splines_scale = (scale1 + scale2) / 2.0

        # Use scaled tolerance for intersection detection
        scaled_tolerance = tolerance * splines_scale

        results = IntersectBSplines(spline1, spline2, scaled_tolerance)

        # Use the IntersectBSplines class to find intersections
        return [(r["parmOnCurve1"], r["parmOnCurve2"]) for r in results]

    @staticmethod
    def _scale_curve(spline: Geom_BSplineCurve) -> float:
        """
        Returns the approximate scale of a single B-spline curve.
        Matches the C++ scale(const Handle(Geom_BSplineCurve)& spline) function.
        """
        scale_val = 0.0
        if spline.NbPoles() > 0:
            first_ctrl_pnt = spline.Pole(1)
            for ctrl_pnt_idx in range(2, spline.NbPoles() + 1):
                distance = first_ctrl_pnt.Distance(spline.Pole(ctrl_pnt_idx))
                scale_val = max(scale_val, distance)
        return scale_val if scale_val > 0 else 1.0

    @staticmethod
    def _scale_curve_list(splines_vector: list[Geom_BSplineCurve]) -> float:
        """
        Returns the approximate scale of the biggest B-spline curve in a list.
        Matches the C++ scale(const std::vector<Handle(Geom_BSplineCurve)>& splines_vector) function.
        """
        max_scale = 0.0
        for spline in splines_vector:
            max_scale = max(BSplineAlgorithms._scale_curve(spline), max_scale)
        return max_scale if max_scale > 0 else 1.0

    @staticmethod
    def _scale_array2_of_pnt(points: TColgp_Array2OfPnt) -> float:
        """
        Returns the scale of the point matrix.
        Matches the C++ scale(const TColgp_Array2OfPnt& points) function.
        """
        the_scale = 0.0
        for u_idx in range(points.LowerRow(), points.UpperRow() + 1):
            p_first = points(u_idx, points.LowerCol())
            for v_idx in range(points.LowerCol() + 1, points.UpperCol() + 1):
                dist = p_first.Distance(points(u_idx, v_idx))
                the_scale = max(the_scale, dist)
        return the_scale

    @staticmethod
    def _scale_array1_of_pnt(points: TColgp_Array1OfPnt) -> float:
        """
        Returns the scale of the point list by searching for the largest distance between two points.
        Matches the C++ scale(const TColgp_Array1OfPnt& points) function.
        """
        the_scale = 0.0
        for i in range(points.Lower(), points.Upper() + 1):
            for j in range(
                i + 1, points.Upper() + 1
            ):  # C++ iterates from i+1 to Upper()-1, but Upper() is inclusive
                dist = points(i).Distance(points(j))
                the_scale = max(the_scale, dist)
        return the_scale if the_scale > 0 else 1.0  # Ensure non-zero scale

    @staticmethod
    def scale(
        obj: (
            Geom_BSplineCurve
            | list[Geom_BSplineCurve]
            | TColgp_Array2OfPnt
            | TColgp_Array1OfPnt
        ),
    ) -> float:
        """
        Returns the approximate scale of the given B-spline curve(s) or point array(s).
        This is a unified method to match the overloaded C++ scale functions.
        """
        if isinstance(obj, Geom_BSplineCurve):
            return BSplineAlgorithms._scale_curve(obj)
        elif isinstance(obj, list):  # Assuming list of Geom_BSplineCurve
            return BSplineAlgorithms._scale_curve_list(obj)
        elif isinstance(obj, TColgp_Array2OfPnt):
            return BSplineAlgorithms._scale_array2_of_pnt(obj)
        elif isinstance(obj, TColgp_Array1OfPnt):
            return BSplineAlgorithms._scale_array1_of_pnt(obj)
        else:
            raise TypeError("Unsupported type for scale function")

    # Helper methods
    @staticmethod
    def _get_knots(spline: Geom_BSplineCurve) -> list[float]:
        """Get all knots from a B-spline curve."""
        knots = []
        n_knots = spline.NbKnots()
        for i in range(1, n_knots + 1):
            knots.append(spline.Knot(i))
        return knots

    @staticmethod
    def _get_knot_multiplicity(
        spline: Geom_BSplineCurve, knot: float, tol: float
    ) -> int:
        """Get multiplicity of a specific knot."""
        n_knots = spline.NbKnots()
        for i in range(1, n_knots + 1):
            if abs(spline.Knot(i) - knot) < tol:
                return spline.Multiplicity(i)
        return 0

    @staticmethod
    def _get_poles(spline: Geom_BSplineCurve) -> list[gp_Pnt]:
        """Get all poles from a B-spline curve."""
        poles = []
        n_poles = spline.NbPoles()
        for i in range(1, n_poles + 1):
            poles.append(spline.Pole(i))
        return poles

    @staticmethod
    def _get_weights(spline: Geom_BSplineCurve) -> list[float]:
        """Get all weights from a rational B-spline curve."""
        if not spline.IsRational():
            return [1.0] * spline.NbPoles()

        weights = []
        n_poles = spline.NbPoles()
        for i in range(1, n_poles + 1):
            weights.append(spline.Weight(i))
        return weights

    @staticmethod
    def _insert_knots(
        spline: Geom_BSplineCurve,
        knots: list[float],
        multiplicities: list[int],
        tol: float,
    ) -> Geom_BSplineCurve:
        """
        Insert knots into a B-spline curve.

        Args:
            spline: Original B-spline curve
            knots: Knots to insert
            multiplicities: Multiplicities for each knot
            tol: Tolerance for knot comparison

        Returns:
            New B-spline with inserted knots
        """

        # Create new spline
        new_spline = clone_bspline(spline)

        # Insert each knot with its specified multiplicity
        for knot, mult in zip(knots, multiplicities):
            # Check if knot already exists within tolerance
            knot_exists = False
            for i in range(1, new_spline.NbKnots() + 1):
                if abs(new_spline.Knot(i) - knot) < tol:
                    knot_exists = True
                    new_spline.IncreaseMultiplicity(i, mult)
                    # current_mult = new_spline.Multiplicity(i)
                    # if current_mult < mult:
                    #     # Increase multiplicity to required level
                    #     new_spline.InsertKnot(knot, mult - current_mult, tol, False)
                    break

            if not knot_exists:
                # Insert new knot
                new_spline.InsertKnot(knot, mult, tol, False)

        return new_spline

    @staticmethod
    def reparametrize_bspline_continuously_approx(
        spline: Geom_BSplineCurve,
        old_parameters: list[float],
        new_parameters: list[float],
        n_control_pnts: int,
    ) -> ApproxResult:  # Note: C++ returns ApproxResult
        """
        Reparametrize B-spline curve using approximation, matching C++ line by line.

        Args:
            spline: Original B-spline curve
            old_parameters: Original parameter values at intersection points
            new_parameters: New parameter values at intersection points
            n_control_pnts: Maximum number of control points for the new curve

        Returns:
            ApproxResult containing the reparametrized curve and other info.
        """

        # Helper for IsInsideTolerance
        def _is_inside_tolerance(value, target, tolerance):
            return abs(value - target) < tolerance

        # Helper for finding index with tolerance
        def _find_index_with_tolerance(data_list, target, tolerance):
            for i, val in enumerate(data_list):
                if abs(val - target) < tolerance:
                    return i
            return -1

        if len(old_parameters) != len(new_parameters):
            raise error("parameter sizes dont match")

        # create a B-spline as a function for reparametrization
        old_parameters_pnts = TColgp_HArray1OfPnt2d(1, len(old_parameters))
        for parameter_idx in range(len(old_parameters)):
            occIdx = parameter_idx + 1
            old_parameters_pnts.SetValue(
                occIdx, gp_Pnt2d(old_parameters[parameter_idx], 0)
            )

        # Convert new_parameters (List[float]) to TColStd_Array1OfReal
        new_parameters_array = BSplineAlgorithms.to_array(new_parameters)

        # Correct order of arguments: old_parameters_pnts, PeriodicFlag, new_parameters_array
        interpolationObject = Geom2dAPI_Interpolate(
            old_parameters_pnts, new_parameters_array, False, 1e-6
        )

        interpolationObject.Perform()

        if not interpolationObject.IsDone():
            raise error("Cannot reparametrize", ErrorCode.MATH_ERROR)

        reparametrizing_spline = (
            interpolationObject.Curve()
        )  # This returns Geom2d_BSplineCurve

        breaks = []
        for ipar in range(1, len(new_parameters) - 1):
            breaks.append(new_parameters[ipar])

        par_tol = 1e-10

        # MODEL_KINKS section
        kinks = BSplineAlgorithms.get_kink_parameters(spline)

        for ikink in range(len(kinks)):
            projected_param = Geom2dAPI_ProjectPointOnCurve(
                gp_Pnt2d(kinks[ikink], 0.0), reparametrizing_spline
            ).LowerDistanceParameter()
            kinks[ikink] = projected_param

        new_breaks = []
        for b in breaks:
            is_kink_nearby = False
            for k in kinks:
                if _is_inside_tolerance(b, k, par_tol):
                    is_kink_nearby = True
                    break
            if not is_kink_nearby:
                new_breaks.append(b)
        breaks = new_breaks

        parameters = BSplineAlgorithms.linspace_with_breaks(
            new_parameters[0], new_parameters[-1], max(101, n_control_pnts * 2), breaks
        )

        for kink in kinks:
            bisect.insort_left(parameters, kink)

        points = TColgp_HArray1OfPnt(1, len(parameters))
        for i in range(1, len(parameters) + 1):
            oldParameter = reparametrizing_spline.Value(parameters[i - 1]).X()
            points.SetValue(i, spline.Value(oldParameter))

        makeContinuous = spline.IsClosed() and (
            spline.DN(spline.FirstParameter(), 1).Angle(
                spline.DN(spline.LastParameter(), 1)
            )
            < 6.0 / 180.0 * math.pi
        )

        from .bspline_approx_interp import BSplineApproxInterp

        approximationObj = BSplineApproxInterp(
            points, int(n_control_pnts), 3, makeContinuous
        )

        breaks_for_interpolation = []
        breaks_for_interpolation.append(new_parameters[0])
        breaks_for_interpolation.extend(breaks)
        breaks_for_interpolation.append(new_parameters[-1])

        for thebreak in breaks_for_interpolation:
            idx = _find_index_with_tolerance(parameters, thebreak, par_tol)
            if idx != -1:
                approximationObj.interpolate_point(idx)

        for kink in kinks:
            idx = _find_index_with_tolerance(parameters, kink, par_tol)
            if idx != -1:
                approximationObj.interpolate_point(idx, True)

        result = approximationObj.fit_curve_optimal(parameters)

        assert result.curve is not None

        return result

    @staticmethod
    def reparametrize_bspline_simple(
        spline: Geom_BSplineCurve, umin: float, umax: float
    ) -> Geom_BSplineCurve:
        """
        Simple linear reparameterization of B-spline curve.

        Args:
            spline: Original B-spline curve
            umin: New minimum parameter
            umax: New maximum parameter

        Returns:
            Reparametrized B-spline curve
        """
        # Create a copy with new parameter range
        current_umin = spline.FirstParameter()
        current_umax = spline.LastParameter()

        # Linear transformation: u_new = a * u_old + b
        a = (umax - umin) / (current_umax - current_umin)
        b = umin - a * current_umin

        # Get current knots
        n_knots = spline.NbKnots()
        knots_array = TColStd_Array1OfReal(1, n_knots)
        for i in range(1, n_knots + 1):
            old_knot = spline.Knot(i)
            new_knot = a * old_knot + b
            knots_array.SetValue(i, new_knot)

        # Create new spline with transformed knots
        n_poles = spline.NbPoles()
        poles = TColgp_Array1OfPnt(1, n_poles)
        weights = TColStd_Array1OfReal(1, n_poles)

        for i in range(1, n_poles + 1):
            poles.SetValue(i, spline.Pole(i))
            weights.SetValue(i, spline.Weight(i))

        mult_array = TColStd_Array1OfInteger(1, n_knots)
        for i in range(1, n_knots + 1):
            mult_array.SetValue(i, spline.Multiplicity(i))

        return Geom_BSplineCurve(
            poles,
            weights,
            knots_array,
            mult_array,
            spline.Degree(),
            spline.IsPeriodic(),
        )

    @staticmethod
    def _convert_to_bspline(curve: Geom_Curve) -> Geom_BSplineCurve:
        """
        Convert generic curve to B-spline curve.

        Args:
            curve: Generic curve to convert

        Returns:
            B-spline curve
        """
        # Convert using OCP GeomConvert utility
        return GeomConvert.CurveToBSplineCurve_s(curve)

    @staticmethod
    def knots_from_curve_parameters(
        params: list[float], degree: int, closed_curve: bool = False
    ) -> list[float]:
        """
        Create knot vector from curve parameters following Park (2000) algorithm.

        Args:
            params: Parameter values (corresponding to control points)
            degree: Degree of the spline
            closed_curve: Whether the curve is closed

        Returns:
            Knot vector
        """
        if len(params) < 2:
            raise ValueError("Parameters must contain two or more elements.")

        n_cp = len(params)
        if closed_curve:
            # For each continuity condition, we have to add one control point
            n_cp += degree - 1

        n_inner_knots = n_cp - degree + 1

        inner_knots = [0.0] * n_inner_knots
        inner_knots[0] = params[0]
        inner_knots[-1] = params[-1]

        knots = []

        if closed_curve and degree % 2 == 0:
            m = len(params) - 2

            # Build difference vector
            dparm = [0.0] * (m + 1)
            for iparm in range(m + 1):
                dparm[iparm] = params[iparm + 1] - params[iparm]

            inner_knots[1] = inner_knots[0] + 0.5 * (dparm[0] + dparm[m])
            for iparm in range(1, m):
                inner_knots[iparm + 1] = inner_knots[iparm] + 0.5 * (
                    dparm[iparm - 1] + dparm[iparm]
                )

            # Shift parameters
            for iparm in range(len(params)):
                params[iparm] += dparm[m] / 2.0

        elif closed_curve:
            if len(inner_knots) != len(params):
                raise ValueError(
                    "Inner knots size must match parameters size for closed curves"
                )
            inner_knots = params.copy()
        else:
            # Averaging method for open curves
            for j in range(1, len(params) - degree):
                sum_val = 0.0
                # Average
                for i in range(j, j + degree):
                    sum_val += params[i]
                inner_knots[j] = sum_val / float(degree)

        if closed_curve:
            offset = inner_knots[0] - inner_knots[n_inner_knots - 1]
            for iknot in range(degree):
                knots.append(offset + inner_knots[n_inner_knots - degree - 1 + iknot])
            for iknot in range(n_inner_knots):
                knots.append(inner_knots[iknot])
            for iknot in range(degree):
                knots.append(-offset + inner_knots[iknot + 1])
        else:
            for iknot in range(degree):
                knots.append(inner_knots[0])
            for iknot in range(n_inner_knots):
                knots.append(inner_knots[iknot])
            for iknot in range(degree):
                knots.append(inner_knots[n_inner_knots - 1])

        if closed_curve and degree <= 1:
            n_knots = len(knots)
            knots[0] = knots[1]
            knots[n_knots - 1] = knots[n_knots - 2]

        return knots

    @staticmethod
    def bspline_basis_mat(
        degree: int,
        knots: TColStd_Array1OfReal,
        params: TColStd_Array1OfReal,
        deriv_order: int = 0,
    ) -> np.ndarray:
        """
        Compute B-spline basis matrix.

        Args:
            degree: Degree of B-spline
            knots: Knot vector
            params: Parameter values
            deriv_order: Derivative order (0 = basis functions)

        Returns:
            Basis matrix as numpy array
        """
        # This is a simplified implementation - in C++ this uses BSplCLib
        # For now, we'll implement a basic version
        n_cp = knots.Length() - degree - 1
        n_params = params.Length()

        # Create numpy matrix
        basis_mat = np.zeros((n_params, n_cp))

        # For each parameter, compute basis functions
        for i_param in range(1, n_params + 1):
            param_val = params(i_param)

            # Find span index
            span_idx = BSplineAlgorithms._find_span(knots, param_val, degree)

            # Compute basis functions
            basis_functions = BSplineAlgorithms._basis_functions(
                knots, span_idx, param_val, degree
            )

            # Fill matrix row
            for j in range(degree + 1):
                # The control points involved are P_{span_idx - degree - 1} to P_{span_idx - 1} (0-indexed)
                col_idx = span_idx - degree - 1 + j
                if 0 <= col_idx < n_cp:  # Ensure index is within bounds
                    basis_mat[i_param - 1, col_idx] = basis_functions[j]

        return basis_mat

    @staticmethod
    def _find_span(knots: TColStd_Array1OfReal, param: float, degree: int) -> int:
        """
        Find the span index for a given parameter (1-indexed).
        Algorithm A2.1 from The NURBS Book.
        """
        n_cp = knots.Length() - degree - 1  # Number of control points (0-indexed count)

        # Special case for parameter at end
        # If param is the last knot, the span index is n_cp (1-indexed)
        if param >= knots(n_cp + 1):  # knots(n_cp + 1) is the last knot value
            return n_cp  # This is the 1-indexed span index

        # Binary search for span index k such that knots(k) <= param < knots(k+1)
        # The span index k ranges from degree to n_cp (1-indexed)
        low = degree  # 1-indexed
        high = n_cp + 1  # 1-indexed

        while high - low > 1:
            mid = (low + high) // 2
            if param < knots(mid):
                high = mid
            else:
                low = mid
        return low

    @staticmethod
    def _basis_functions(
        knots: TColStd_Array1OfReal, span_idx: int, param: float, degree: int
    ) -> list[float]:
        """Compute B-spline basis functions for a given span."""
        left = [0.0] * (degree + 1)
        right = [0.0] * (degree + 1)
        N = [0.0] * (degree + 1)

        N[0] = 1.0

        for j in range(1, degree + 1):
            left[j] = param - knots(span_idx + 1 - j)
            right[j] = knots(span_idx + j) - param
            saved = 0.0

            for r in range(j):
                temp = N[r] / (right[r + 1] + left[j - r])
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp

            N[j] = saved

        return N

    @staticmethod
    def max_distance_of_bounding_box(points: TColgp_Array1OfPnt) -> float:
        """
        Compute maximum distance between points in bounding box.

        Args:
            points: Array of points

        Returns:
            Maximum distance
        """
        max_distance = 0.0
        for i in range(points.Lower(), points.Upper() + 1):
            for j in range(points.Lower(), points.Upper() + 1):
                dist = points(i).Distance(points(j))
                max_distance = max(max_distance, dist)
        return max_distance

    @staticmethod
    def is_closed(points: TColgp_HArray1OfPnt, c2_continuous: bool = False) -> bool:
        """
        Check if points form a closed curve.

        Args:
            points: Points to check
            c2_continuous: Whether to require C2 continuity

        Returns:
            True if curve is closed
        """
        max_distance = BSplineAlgorithms.max_distance_of_bounding_box(points.Array1())
        error = 1e-6 * max_distance
        return (
            points(points.Lower()).IsEqual(points(points.Upper()), error)
            and c2_continuous
        )

    @staticmethod
    def to_array(vector: list[float]) -> TColStd_HArray1OfReal:
        """
        Convert Python list to TColStd_HArray1OfReal.

        Args:
            vector: List of values

        Returns:
            TColStd_HArray1OfReal handle
        """
        array = TColStd_HArray1OfReal(1, len(vector))
        for i, value in enumerate(vector, 1):
            array.SetValue(i, value)
        return array

    @staticmethod
    def to_array_int(vector: list[int]) -> TColStd_HArray1OfInteger:
        """
        Convert Python list to TColStd_HArray1OfInteger.

        Args:
            vector: List of integer values

        Returns:
            TColStd_HArray1OfInteger handle
        """
        array = TColStd_HArray1OfInteger(1, len(vector))
        for i, value in enumerate(vector, 1):
            array.SetValue(i, value)
        return array

    @staticmethod
    def _pnt_array2_get_column(
        matrix: TColgp_Array2OfPnt, col_index: int
    ) -> TColgp_HArray1OfPnt:
        """
        Extracts a column from a TColgp_Array2OfPnt and returns it as a TColgp_HArray1OfPnt.
        Matches the C++ array2GetColumn helper function.
        """
        lower_row = matrix.LowerRow()
        upper_row = matrix.UpperRow()
        col_vector = TColgp_HArray1OfPnt(lower_row, upper_row)

        for row_idx in range(lower_row, upper_row + 1):
            col_vector.SetValue(row_idx, matrix(row_idx, col_index))
        return col_vector

    @staticmethod
    def _pnt_array2_get_row(
        matrix: TColgp_Array2OfPnt, row_index: int
    ) -> TColgp_HArray1OfPnt:
        """
        Extracts a row from a TColgp_Array2OfPnt and returns it as a TColgp_HArray1OfPnt.
        Matches the C++ array2GetRow helper function.
        """
        lower_col = matrix.LowerCol()
        upper_col = matrix.UpperCol()
        row_vector = TColgp_HArray1OfPnt(lower_col, upper_col)

        for col_idx in range(lower_col, upper_col + 1):
            row_vector.SetValue(col_idx, matrix(row_index, col_idx))
        return row_vector

    @staticmethod
    def compute_params_bspline_surf(
        points: TColgp_Array2OfPnt, alpha: float = 0.5
    ) -> tuple[list[float], list[float]]:
        """
        Computes parameters for a B-spline surface in both u and v directions.
        Matches the C++ computeParamsBSplineSurf function line by line.
        """
        # First for parameters in u-direction:
        # points.ColLength() gives the number of columns (u-direction control points)
        # points.RowLength() gives the number of rows (v-direction control points)

        # paramsU will store the averaged parameters for the u-direction.
        # Its size should be equal to the number of control points in the u-direction (matrix.ColLength())
        params_u = [0.0] * points.ColLength()

        # Iterate over each column (v-index) to compute u-parameters for that "profile"
        for v_idx in range(points.LowerCol(), points.UpperCol() + 1):
            # Extract the column as a 1D array of points (TColgp_HArray1OfPnt)
            points_u_line = BSplineAlgorithms._pnt_array2_get_column(points, v_idx)

            # Compute parameters for this 1D array of points
            parameters_u_line = BSplineAlgorithms.compute_params_bspline_curve(
                points_u_line, alpha
            )

            # Average over columns: sum up the parameters
            for u_param_idx in range(len(parameters_u_line)):
                params_u[u_param_idx] += parameters_u_line[u_param_idx]

        # Divide by the number of rows (v-direction control points) to get the average
        for u_param_idx in range(len(params_u)):
            params_u[u_param_idx] /= float(points.RowLength())

        # Now for parameters in v-direction:
        # paramsV will store the averaged parameters for the v-direction.
        # Its size should be equal to the number of control points in the v-direction (matrix.RowLength())
        params_v = [0.0] * points.RowLength()

        # Iterate over each row (u-index) to compute v-parameters for that "guide"
        for u_idx in range(points.LowerRow(), points.UpperRow() + 1):
            # Extract the row as a 1D array of points (TColgp_HArray1OfPnt)
            points_v_line = BSplineAlgorithms._pnt_array2_get_row(points, u_idx)

            # Compute parameters for this 1D array of points
            parameters_v_line = BSplineAlgorithms.compute_params_bspline_curve(
                points_v_line, alpha
            )

            # Average over rows: sum up the parameters
            for v_param_idx in range(len(parameters_v_line)):
                params_v[v_param_idx] += parameters_v_line[v_param_idx]

        # Divide by the number of columns (u-direction control points) to get the average
        for v_param_idx in range(len(params_v)):
            params_v[v_param_idx] /= float(points.ColLength())

        # Return computed parameters for both u- and v-direction
        return params_u, params_v

    @staticmethod
    def flip_surface(surface: Geom_BSplineSurface) -> Geom_BSplineSurface:
        """
        Swaps axes of the given surface, i.e., surface(u-coord, v-coord) becomes surface(v-coord, u-coord).
        Matches the C++ flipSurface function.
        """
        result = clone_bspline_surface(surface)
        result.ExchangeUV()
        return result

    @staticmethod
    def _insert_knot_with_multiplicity(
        knot: float,
        count: int,
        degree: int,
        knots: list[float],
        mults: list[int],
        tol: float = 1e-5,
    ):
        """
        Inserts a knot into the knot vector with a specified multiplicity,
        similar to the C++ insertKnot helper function.

        Args:
            knot: The knot value to insert.
            count: The desired multiplicity for the knot.
            degree: The degree of the B-spline.
            knots: The list of distinct knot values (modified in place).
            mults: The list of multiplicities for each distinct knot (modified in place).
            tol: Tolerance for knot comparison.
        """
        if not (knots[0] - tol <= knot <= knots[-1] + tol):
            raise error("knot out of range", ErrorCode.INVALID_ARGUMENT)

        # Find if knot already exists within tolerance
        found_pos = -1
        for i, k in enumerate(knots):
            if abs(k - knot) < tol:
                found_pos = i
                break

        if found_pos != -1:
            # Knot found, increase multiplicity up to the degree
            mults[found_pos] = min(mults[found_pos] + count, degree)
        else:
            # Knot not found, insert new one maintaining sorted order
            insert_idx = 0
            while insert_idx < len(knots) and knots[insert_idx] < knot:
                insert_idx += 1
            knots.insert(insert_idx, knot)
            mults.insert(
                insert_idx, min(count, degree)
            )  # Multiplicity should not exceed degree

    class SurfAdapterView:
        """
        Adapter class for Geom_BSplineSurface to provide a unified interface
        for knot vector manipulation, similar to C++'s SurfAdapterView.
        """

        def __init__(self, surf: Geom_BSplineSurface, direction: "SurfaceDirection"):
            self._surf = surf
            self._dir = direction

        def insert_knot(self, knot: float, mult: int, tolerance: float = 1e-15):
            if self._dir == SurfaceDirection.u:
                self._surf.InsertUKnot(knot, mult, tolerance, False)
            else:  # SurfaceDirection.v
                self._surf.InsertVKnot(knot, mult, tolerance, False)

        def get_knot(self, idx: int) -> float:
            if self._dir == SurfaceDirection.u:
                return self._surf.UKnot(idx)
            else:  # SurfaceDirection.v
                return self._surf.VKnot(idx)

        def get_mult(self, idx: int) -> int:
            if self._dir == SurfaceDirection.u:
                return self._surf.UMultiplicity(idx)
            else:  # SurfaceDirection.v
                return self._surf.VMultiplicity(idx)

        def get_n_knots(self) -> int:
            if self._dir == SurfaceDirection.u:
                return self._surf.NbUKnots()
            else:  # SurfaceDirection.v
                return self._surf.NbVKnots()

        def get_degree(self) -> int:
            if self._dir == SurfaceDirection.u:
                return self._surf.UDegree()
            else:  # SurfaceDirection.v
                return self._surf.VDegree()

        def set_dir(self, direction: "SurfaceDirection"):
            self._dir = direction

        def get_surface(self) -> Geom_BSplineSurface:
            return self._surf

    class CurveAdapterView:
        """
        Adapter class for Geom_BSplineCurve to provide a unified interface
        for knot vector manipulation, similar to C++'s CurveAdapterView.
        """

        def __init__(self, curve: Geom_BSplineCurve):
            self._curve = curve

        def insert_knot(self, knot: float, mult: int, tolerance: float = 1e-15):
            self._curve.InsertKnot(knot, mult, tolerance, False)

        def get_knot(self, idx: int) -> float:
            return self._curve.Knot(idx)

        def get_mult(self, idx: int) -> int:
            return self._curve.Multiplicity(idx)

        def get_n_knots(self) -> int:
            return self._curve.NbKnots()

        def get_degree(self) -> int:
            return self._curve.Degree()

        def get_curve(self) -> Geom_BSplineCurve:
            return self._curve

    @staticmethod
    def have_same_range(
        splines_vector: list[
            Union[
                "BSplineAlgorithms.SurfAdapterView",
                "BSplineAlgorithms.CurveAdapterView",
            ]
        ],
        par_tolerance: float,
    ) -> bool:
        """
        Checks if all splines in the vector have the same parameter range.
        This is a generic helper for both curves and surfaces (via adapters).
        """
        if not splines_vector:
            return True

        first_adapter = splines_vector[0]
        # Ensure first_adapter is an adapter type
        if not isinstance(
            first_adapter,
            (BSplineAlgorithms.SurfAdapterView, BSplineAlgorithms.CurveAdapterView),
        ):
            raise TypeError(
                "Unsupported spline type in have_same_range: first element is not an adapter."
            )

        begin_param_ref = first_adapter.get_knot(1)
        end_param_ref = first_adapter.get_knot(first_adapter.get_n_knots())

        for spline_idx in range(1, len(splines_vector)):
            current_adapter = splines_vector[spline_idx]
            if not isinstance(
                current_adapter,
                (BSplineAlgorithms.SurfAdapterView, BSplineAlgorithms.CurveAdapterView),
            ):
                raise TypeError(
                    "Mixed types in splines_vector for have_same_range: element is not an adapter."
                )

            begin_param_current = current_adapter.get_knot(1)
            end_param_current = current_adapter.get_knot(current_adapter.get_n_knots())

            if (
                abs(begin_param_current - begin_param_ref) > par_tolerance
                or abs(end_param_current - end_param_ref) > par_tolerance
            ):
                return False
        return True
