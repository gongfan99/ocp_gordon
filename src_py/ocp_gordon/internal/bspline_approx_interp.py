"""
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2018 German Aerospace Center (DLR)

Created: 2019-04-24 Martin Siggel <Martin.Siggel@dlr.de>
"""

import math
from typing import List, Optional, Tuple

import numpy as np
from OCP.BSplCLib import BSplCLib  # For KnotSequence
from OCP.Geom import Geom_BSplineCurve, Geom_Curve
from OCP.GeomAPI import (
    GeomAPI_ProjectPointOnCurve,  # Equivalent to Geom2dAPI_ProjectPointOnCurve for 3D
)
from OCP.gp import gp_Pnt, gp_Vec
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.TColStd import (
    TColStd_Array1OfInteger,
    TColStd_Array1OfReal,
    TColStd_HArray1OfInteger,
    TColStd_HArray1OfReal,
)

from .approx_result import ApproxResult
from .bspline_algorithms import BSplineAlgorithms  # For bspline_basis_mat, scale, etc.
from .error import ErrorCode, error


class ProjectResult:
    """
    Structure to hold the result of projecting a point onto a curve.
    """

    def __init__(self, parameter: float, error: float):
        self.parameter = parameter
        self.error = error


class BSplineApproxInterp:
    """
    Approximation and interpolation of B-spline curves.
    """

    def __init__(
        self,
        points: TColgp_Array1OfPnt,
        n_control_points: int,
        degree: int = 3,
        continuous_if_closed: bool = False,
    ):
        self.m_pnts = TColgp_Array1OfPnt(1, points.Length())
        for i in range(points.Lower(), points.Upper() + 1):
            self.m_pnts.SetValue(i, points(i))

        self.m_index_of_approximated: list[int] = list(range(points.Length()))
        self.m_index_of_interpolated: list[int] = []
        self.m_index_of_kinks: list[int] = []

        self.m_degree = degree
        self.m_ncp = n_control_points
        self.m_c2_continuous = continuous_if_closed

    def interpolate_point(self, point_index: int, with_kink: bool = False):
        if point_index not in self.m_index_of_approximated:
            raise error(
                "Invalid index in BSplineApproxInterp::interpolate_point",
                ErrorCode.INDEX_ERROR,
            )

        self.m_index_of_approximated.remove(point_index)
        self.m_index_of_interpolated.append(point_index)
        if with_kink:
            self.m_index_of_kinks.append(point_index)

    def _max_distance_of_bounding_box(self, points: TColgp_Array1OfPnt) -> float:
        if points.Length() == 0:
            return 0.0

        np_array = np.zeros((points.Length(), 3))
        for i in range(points.Lower(), points.Upper() + 1):
            p = points(i)
            np_array[i - points.Lower()] = [p.X(), p.Y(), p.Z()]
        low = np.min(np_array, axis=0)
        high = np.max(np_array, axis=0)
        delta = high - low
        return math.sqrt(np.dot(delta, delta))

        # min_p = gp_Pnt(float("inf"), float("inf"), float("inf"))
        # max_p = gp_Pnt(float("-inf"), float("-inf"), float("-inf"))

        # for i in range(points.Lower(), points.Upper() + 1):
        #     p = points(i)
        #     min_p.SetX(min(min_p.X(), p.X()))
        #     min_p.SetY(min(min_p.Y(), p.Y()))
        #     min_p.SetZ(min(min_p.Z(), p.Z()))

        #     max_p.SetX(max(max_p.X(), p.X()))
        #     max_p.SetY(max(max_p.Y(), p.Y()))
        #     max_p.SetZ(max(max_p.Z(), p.Z()))

        # return max_p.Distance(min_p)

    def is_closed(self) -> bool:
        if not self.m_c2_continuous:
            return False
        max_distance = self._max_distance_of_bounding_box(self.m_pnts)
        error_val = 1e-12 * max_distance
        return self.m_pnts(self.m_pnts.Lower()).IsEqual(
            self.m_pnts(self.m_pnts.Upper()), error_val
        )

    def _first_and_last_interpolated(self) -> bool:
        first = 0 in self.m_index_of_interpolated
        last = (self.m_pnts.Length() - 1) in self.m_index_of_interpolated
        return first and last

    def _compute_parameters(self, alpha: float) -> list[float]:
        sum_len = 0.0
        n_points = self.m_pnts.Length()
        params = [0.0] * n_points

        # calc total arc length: dt^2 = dx^2 + dy^2
        for i in range(1, n_points):
            p1 = self.m_pnts(self.m_pnts.Lower() + i - 1)
            p2 = self.m_pnts(self.m_pnts.Lower() + i)
            len2 = p1.SquareDistance(p2)
            sum_len += math.pow(len2, alpha / 2.0)
            params[i] = sum_len

        # normalize parameter with maximum
        tmax = params[n_points - 1]
        if tmax < 1e-10:  # Handle case where all points are identical
            for i in range(n_points):
                params[i] = float(i) / (n_points - 1) if n_points > 1 else 0.0
        else:
            for i in range(1, n_points):
                params[i] /= tmax

        # reset end value to achieve a better accuracy
        if n_points > 0:
            params[n_points - 1] = 1.0
        return params

    def _compute_knots(
        self, ncp: int, params: list[float]
    ) -> tuple[list[float], list[int]]:
        order = self.m_degree + 1
        if ncp < order:
            raise error("Number of control points too small!", ErrorCode.MATH_ERROR)

        umin = min(params)
        umax = max(params)

        knots: list[float] = [0.0] * (ncp - self.m_degree + 1)
        mults: list[int] = [0] * (ncp - self.m_degree + 1)

        # fill multiplicity at start
        knots[0] = umin
        mults[0] = order

        # number of knots between the multiplicities
        N = ncp - order
        # set uniform knot distribution
        for i in range(1, N + 1):
            knots[i] = umin + (umax - umin) * float(i) / float(N + 1)
            mults[i] = 1

        # fill multiplicity at end
        knots[N + 1] = umax
        mults[N + 1] = order

        for kink_idx in self.m_index_of_kinks:
            knot_val = params[kink_idx]
            BSplineAlgorithms._insert_knot_with_multiplicity(
                knot_val, self.m_degree, self.m_degree, knots, mults, 1e-4
            )

        return knots, mults

    def fit_curve(self, initial_params: list[float] | None = None) -> ApproxResult:
        params = (
            initial_params
            if initial_params is not None and initial_params
            else self._compute_parameters(0.5)
        )

        if len(params) != self.m_pnts.Length():
            raise error("Number of parameters don't match number of points")

        knots_list, mults_list = self._compute_knots(self.m_ncp, params)

        # Convert to OCP arrays
        knots_array = TColStd_Array1OfReal(1, len(knots_list))
        for i, k in enumerate(knots_list, 1):
            knots_array.SetValue(i, k)

        mults_array = TColStd_Array1OfInteger(1, len(mults_list))
        for i, m in enumerate(mults_list, 1):
            mults_array.SetValue(i, m)

        return self._solve(params, knots_array, mults_array)

    def fit_curve_optimal(
        self, initial_params: list[float] | None = None, max_iter: int = 10
    ) -> ApproxResult:
        params = (
            initial_params
            if initial_params is not None and initial_params
            else self._compute_parameters(0.5)
        )

        if len(params) != self.m_pnts.Length():
            raise error("Number of parameters don't match number of points")

        knots_list, mults_list = self._compute_knots(self.m_ncp, params)

        knots_array = TColStd_Array1OfReal(1, len(knots_list))
        for i, k in enumerate(knots_list, 1):
            knots_array.SetValue(i, k)

        mults_array = TColStd_Array1OfInteger(1, len(mults_list))
        for i, m in enumerate(mults_list, 1):
            mults_array.SetValue(i, m)

        iteration = 0
        result = self._solve(params, knots_array, mults_array)
        old_error = result.error * 2.0

        while (
            result.error > 0
            and (old_error - result.error) / max(result.error, 1e-6) > 1e-3
            and iteration < max_iter
            and result.curve is not None
        ):
            old_error = result.error
            self._optimize_parameters(result.curve, params)
            result = self._solve(params, knots_array, mults_array)
            iteration += 1

        return result

    def _project_on_curve(
        self, pnt: gp_Pnt, curve: Geom_Curve, initial_param: float
    ) -> ProjectResult:
        max_iter = 10
        eps = 1.0e-6

        t = initial_param

        # Use GeomAPI_ProjectPointOnCurve for robust projection
        projector = GeomAPI_ProjectPointOnCurve(pnt, curve)

        if projector.NbPoints() > 0:
            # Get the parameter of the closest point
            t = projector.LowerDistanceParameter()
            # Calculate the error (distance)
            error_val = pnt.Distance(projector.Point(1))
            return ProjectResult(t, error_val)
        else:
            # Fallback if projection fails, return initial parameter with large error
            return ProjectResult(initial_param, float("inf"))

    def _get_continuity_matrix(
        self,
        n_ctr_pnts: int,
        contin_cons: int,
        params: list[float],
        flat_knots: TColStd_Array1OfReal,
    ) -> np.ndarray:
        continuity_entries = np.zeros((contin_cons, n_ctr_pnts))

        # Need to convert single float to TColStd_Array1OfReal for bspline_basis_mat
        continuity_params1 = TColStd_Array1OfReal(1, 1)
        continuity_params1.SetValue(1, params[0])
        continuity_params2 = TColStd_Array1OfReal(1, 1)
        continuity_params2.SetValue(1, params[len(params) - 1])

        diff1_1 = BSplineAlgorithms.bspline_basis_mat(
            self.m_degree, flat_knots, continuity_params1, 1
        )
        diff1_2 = BSplineAlgorithms.bspline_basis_mat(
            self.m_degree, flat_knots, continuity_params2, 1
        )

        diff2_1 = BSplineAlgorithms.bspline_basis_mat(
            self.m_degree, flat_knots, continuity_params1, 2
        )
        diff2_2 = BSplineAlgorithms.bspline_basis_mat(
            self.m_degree, flat_knots, continuity_params2, 2
        )

        # Set C1 condition (row 0 in 0-indexed numpy array)
        continuity_entries[0, :] = (diff1_1 - diff1_2).flatten()

        # Set C2 condition (row 1)
        continuity_entries[1, :] = (diff2_1 - diff2_2).flatten()

        if not self._first_and_last_interpolated():
            diff0_1 = BSplineAlgorithms.bspline_basis_mat(
                self.m_degree, flat_knots, continuity_params1, 0
            )
            diff0_2 = BSplineAlgorithms.bspline_basis_mat(
                self.m_degree, flat_knots, continuity_params2, 0
            )
            # Set C0 condition (row 2)
            continuity_entries[2, :] = (diff0_1 - diff0_2).flatten()

        return continuity_entries

    def _solve(
        self,
        params: list[float],
        knots: TColStd_Array1OfReal,
        mults: TColStd_Array1OfInteger,
    ) -> ApproxResult:
        # compute flat knots
        n_flat_knots = BSplCLib.KnotSequenceLength_s(mults, self.m_degree, False)
        flat_knots_array = TColStd_Array1OfReal(1, n_flat_knots)
        BSplCLib.KnotSequence_s(knots, mults, flat_knots_array)

        n_approximated = len(self.m_index_of_approximated)
        n_interpolated = len(self.m_index_of_interpolated)
        n_continuity_conditions = 0

        make_closed = self.is_closed()

        if make_closed:
            # C0, C1, C2
            n_continuity_conditions = 3
            if self._first_and_last_interpolated():
                # Remove C0 as they are already equal by design
                n_continuity_conditions -= 1

        # Number of control points required
        n_ctr_pnts = flat_knots_array.Length() - self.m_degree - 1

        if (
            n_ctr_pnts < n_interpolated + n_continuity_conditions
            or n_ctr_pnts < self.m_degree + 1 + n_continuity_conditions
        ):
            raise error("Too few control points for curve interpolation!")

        if (
            n_approximated == 0
            and n_ctr_pnts != n_interpolated + n_continuity_conditions
        ):
            raise error("Wrong number of control points for curve interpolation!")

        # Build left hand side of the equation
        n_vars = n_ctr_pnts + n_interpolated + n_continuity_conditions
        lhs = np.zeros((n_vars, n_vars))

        # Allocate right hand side
        rhs_x = np.zeros(n_vars)
        rhs_y = np.zeros(n_vars)
        rhs_z = np.zeros(n_vars)

        if n_approximated > 0:
            app_params_list = [params[idx] for idx in self.m_index_of_approximated]
            app_params_array = TColStd_Array1OfReal(1, n_approximated)
            for i, p in enumerate(app_params_list, 1):
                app_params_array.SetValue(i, p)

            bx = np.zeros(n_approximated)
            by = np.zeros(n_approximated)
            bz = np.zeros(n_approximated)

            for i, idx in enumerate(self.m_index_of_approximated):
                p = self.m_pnts(self.m_pnts.Lower() + idx)
                bx[i] = p.X()
                by[i] = p.Y()
                bz[i] = p.Z()

            A = BSplineAlgorithms.bspline_basis_mat(
                self.m_degree, flat_knots_array, app_params_array
            )
            At = A.T

            # Add regularization to prevent singular matrix
            lhs[:n_ctr_pnts, :n_ctr_pnts] = At @ A

            rhs_x[:n_ctr_pnts] = At @ bx
            rhs_y[:n_ctr_pnts] = At @ by
            rhs_z[:n_ctr_pnts] = At @ bz

        if n_interpolated + n_continuity_conditions > 0:
            dx = np.zeros(n_interpolated + n_continuity_conditions)
            dy = np.zeros(n_interpolated + n_continuity_conditions)
            dz = np.zeros(n_interpolated + n_continuity_conditions)

            if n_interpolated > 0:
                interp_params_list = [
                    params[idx] for idx in self.m_index_of_interpolated
                ]
                interp_params_array = TColStd_Array1OfReal(1, n_interpolated)
                for i, p in enumerate(interp_params_list, 1):
                    interp_params_array.SetValue(i, p)

                for i, idx in enumerate(self.m_index_of_interpolated):
                    p = self.m_pnts(self.m_pnts.Lower() + idx)
                    dx[i] = p.X()
                    dy[i] = p.Y()
                    dz[i] = p.Z()

                C = BSplineAlgorithms.bspline_basis_mat(
                    self.m_degree, flat_knots_array, interp_params_array
                )

                lhs[:n_ctr_pnts, n_ctr_pnts : n_ctr_pnts + n_interpolated] = (
                    C.T
                )  # C++ uses Ct, but here C is already (n_interpolated, n_ctr_pnts)
                lhs[n_ctr_pnts : n_ctr_pnts + n_interpolated, :n_ctr_pnts] = C

            # sets the C2 continuity constraints for closed curves on the left hand side if requested
            if make_closed:
                continuity_entries = self._get_continuity_matrix(
                    n_ctr_pnts, n_continuity_conditions, params, flat_knots_array
                )

                start_row_lhs = n_ctr_pnts + n_interpolated
                end_row_lhs = start_row_lhs + n_continuity_conditions

                lhs[start_row_lhs:end_row_lhs, :n_ctr_pnts] = continuity_entries
                lhs[:n_ctr_pnts, start_row_lhs:end_row_lhs] = continuity_entries.T

            rhs_x[
                n_ctr_pnts : n_ctr_pnts + n_interpolated + n_continuity_conditions
            ] = dx
            rhs_y[
                n_ctr_pnts : n_ctr_pnts + n_interpolated + n_continuity_conditions
            ] = dy
            rhs_z[
                n_ctr_pnts : n_ctr_pnts + n_interpolated + n_continuity_conditions
            ] = dz

        # Solve linear system
        try:
            # Add regularization to prevent singular matrix
            lhs = lhs + 1e-9 * np.eye(lhs.shape[0])
            cp_x_full = np.linalg.solve(lhs, rhs_x)
            cp_y_full = np.linalg.solve(lhs, rhs_y)
            cp_z_full = np.linalg.solve(lhs, rhs_z)
        except np.linalg.LinAlgError:
            raise error("Singular Matrix", ErrorCode.MATH_ERROR)

        poles = TColgp_Array1OfPnt(1, n_ctr_pnts)
        for i in range(n_ctr_pnts):
            pnt = gp_Pnt(cp_x_full[i], cp_y_full[i], cp_z_full[i])
            poles.SetValue(i + 1, pnt)

        # Create OCP knot and multiplicity arrays for Geom_BSplineCurve constructor
        # Need to convert flat_knots_array back to knots and mults for constructor
        # This is a simplification, as C++ uses the original knots and mults
        # The C++ BSplCLib::KnotSequence is used to get flat_knots, but the constructor
        # for Geom_BSplineCurve takes original knots and mults.
        # For now, we'll use the original knots_list and mults_list from _compute_knots
        # This might need refinement if the flat_knots are truly different.

        # Reconstruct original knots and mults from flat_knots_array if needed,
        # but for now, use the ones computed by _compute_knots

        # The Geom_BSplineCurve constructor expects TColStd_Array1OfReal for knots
        # and TColStd_Array1OfInteger for multiplicities.
        # The `knots` and `mults` parameters passed to `_solve` are already in this format.

        result_curve = Geom_BSplineCurve(poles, knots, mults, self.m_degree, False)

        # compute error
        max_error = 0.0
        for idx in self.m_index_of_approximated:
            ipnt = self.m_pnts.Lower() + idx
            p = self.m_pnts(ipnt)
            par = params[idx]

            error_val = result_curve.Value(par).Distance(p)
            max_error = max(max_error, error_val)

        return ApproxResult(curve=result_curve, error=max_error)

    def _optimize_parameters(self, curve: Geom_Curve, params: list[float]):
        # optimize each parameter by finding it's position on the curve
        for idx in self.m_index_of_approximated:
            pnt_idx = self.m_pnts.Lower() + idx
            res = self._project_on_curve(self.m_pnts(pnt_idx), curve, params[idx])
            params[idx] = res.parameter
