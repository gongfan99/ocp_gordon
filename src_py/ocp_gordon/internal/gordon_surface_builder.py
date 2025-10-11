"""
Gordon surface builder.

This module implements the Gordon surface construction algorithm
that combines skinning surfaces and tensor product surfaces.
"""

import numpy as np
from OCP.Geom import Geom_BSplineCurve, Geom_BSplineSurface
from OCP.TColgp import TColgp_Array2OfPnt
from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCP.gp import gp_Pnt, gp_XYZ
from typing import List, Union, Tuple

from .curves_to_surface import CurvesToSurface
from .bspline_algorithms import BSplineAlgorithms, SurfaceDirection
from .misc import clone_bspline_surface, save_bsplines_to_object
from .error import error, ErrorCode

from .points_to_bspline_interpolation import PointsToBSplineInterpolation


class GordonSurfaceBuilder:
    """
    Builds Gordon surfaces from curve networks.

    The Gordon surface is constructed using the formula:
    S_Gordon(u,v) = S_Profiles(u,v) + S_Guides(u,v) - S_TensorProduct(u,v)
    """

    def __init__(
        self,
        profiles: list[Geom_BSplineCurve],
        guides: list[Geom_BSplineCurve],
        intersection_params_spline_u: list[float],
        intersection_params_spline_v: list[float],
        tolerance: float,
    ):
        """
        Initialize the Gordon surface builder.

        Args:
            profiles: Profile curves (u-direction)
            guides: Guide curves (v-direction)
            intersection_params_spline_u: U parameters at intersection points (length = num_guides)
            intersection_params_spline_v: V parameters at intersection points (length = num_profiles)
            tolerance: Construction tolerance
        """
        self.profiles = profiles
        self.guides = guides
        self.intersection_params_spline_u = intersection_params_spline_u
        self.intersection_params_spline_v = intersection_params_spline_v
        self.tolerance = tolerance

        # Results
        self._surface_profiles = None
        self._surface_guides = None
        self._surface_intersections = None
        self._surface_gordon = None
        self._has_performed = False

    def surface_gordon(self) -> Geom_BSplineSurface:
        """Returns the final Gordon surface."""
        self.perform()
        assert self._surface_gordon is not None
        return self._surface_gordon

    def surface_profiles(self) -> Geom_BSplineSurface:
        """Returns the surface that interpolates the profiles."""
        self.perform()
        assert self._surface_profiles is not None
        return self._surface_profiles

    def surface_guides(self) -> Geom_BSplineSurface:
        """Returns the surface that interpolates the guides."""
        self.perform()
        assert self._surface_guides is not None
        return self._surface_guides

    def surface_intersections(self) -> Geom_BSplineSurface:
        """Returns the surface that interpolates the intersection points."""
        self.perform()
        assert self._surface_intersections is not None
        return self._surface_intersections

    def perform(self):
        """Build all components of the Gordon surface."""
        if self._has_performed:
            return

        self._create_gordon_surface(
            self.profiles,
            self.guides,
            self.intersection_params_spline_u,
            self.intersection_params_spline_v,
        )
        self._has_performed = True

    # In C++ code, points_to_surface and create_common_knots_vector_surface are in BSplineAlgorithms.cpp
    # Due to circular import issues, these two functions are implemented directly in this file instead of being imported from bspline_algorithms.py
    @staticmethod
    def _points_to_surface_internal(
        points: TColgp_Array2OfPnt,
        u_params: list[float],
        v_params: list[float],
        make_u_closed: bool,
        make_v_closed: bool,
    ) -> Geom_BSplineSurface:
        """
        Creates a B-spline surface by interpolating a grid of points.
        Matches the C++ BSplineAlgorithms::pointsToSurface function.
        """
        # 1. Interpolate curves in U-direction (along rows)
        u_direction_curves: list[Geom_BSplineCurve] = []
        for col_idx in range(points.LowerCol(), points.UpperCol() + 1):
            # Extract a row of points as a 1D array
            points_u_line = BSplineAlgorithms._pnt_array2_get_column(points, col_idx)

            # Create a B-spline curve interpolating these points
            interpolator = PointsToBSplineInterpolation(
                points_u_line,
                parameters=u_params,  # Use u_params for curves in u-direction
                continuous_if_closed=make_u_closed,  # Use u-closed flag
            )
            u_direction_curves.append(interpolator.curve())

        # 2. Skin these U-direction curves to form the surface
        # The skinning should happen in the v-direction, interpolating at v_params
        skinner = CurvesToSurface(list(u_direction_curves), v_params, make_v_closed)
        return skinner.surface()

    @staticmethod
    def _create_common_knots_vector_surface_internal(
        surfaces_vector: list[Geom_BSplineSurface],
        direction: SurfaceDirection,
        tol: float = 1e-7,
    ) -> list[Geom_BSplineSurface]:
        """
        Creates common knot vectors for a list of B-spline surfaces in the specified direction(s).
        Matches the C++ BSplineAlgorithms::createCommonKnotsVectorSurface function.
        """
        if not surfaces_vector:
            return []

        result_surfaces = [clone_bspline_surface(s) for s in surfaces_vector]

        if direction == SurfaceDirection.u or direction == SurfaceDirection.both:
            # Collect all unique U-knots and their maximum multiplicities
            all_u_knots = set()
            for surf in result_surfaces:
                for i in range(1, surf.NbUKnots() + 1):
                    all_u_knots.add(surf.UKnot(i))

            sorted_u_knots = sorted(list(all_u_knots))

            common_u_knots_with_mult = []
            for knot in sorted_u_knots:
                max_mult = 0
                for surf in result_surfaces:
                    for i in range(1, surf.NbUKnots() + 1):
                        if abs(surf.UKnot(i) - knot) < tol:
                            current_mult = surf.UMultiplicity(i)
                            max_mult = max(max_mult, current_mult)
                            break  # Found the knot, move to next surface
                common_u_knots_with_mult.append((knot, max_mult))

            # Insert common U-knots into all surfaces
            for surf in result_surfaces:
                for knot, mult in common_u_knots_with_mult:
                    # Check if knot needs to be inserted or multiplicity increased
                    current_mult = 0
                    for i in range(1, surf.NbUKnots() + 1):
                        if abs(surf.UKnot(i) - knot) < tol:
                            current_mult = surf.UMultiplicity(i)
                            break

                    if current_mult < mult:
                        surf.InsertUKnot(knot, mult - current_mult, tol, False)

        if direction == SurfaceDirection.v or direction == SurfaceDirection.both:
            # Collect all unique V-knots and their maximum multiplicities
            all_v_knots = set()
            for surf in result_surfaces:
                for i in range(1, surf.NbVKnots() + 1):
                    all_v_knots.add(surf.VKnot(i))

            sorted_v_knots = sorted(list(all_v_knots))

            common_v_knots_with_mult = []
            for knot in sorted_v_knots:
                max_mult = 0
                for surf in result_surfaces:
                    for i in range(1, surf.NbVKnots() + 1):
                        if abs(surf.VKnot(i) - knot) < tol:
                            current_mult = surf.VMultiplicity(i)
                            max_mult = max(max_mult, current_mult)
                            break  # Found the knot, move to next surface
                common_v_knots_with_mult.append((knot, max_mult))

            # Insert common V-knots into all surfaces
            for surf in result_surfaces:
                for knot, mult in common_v_knots_with_mult:
                    current_mult = 0
                    for i in range(1, surf.NbVKnots() + 1):
                        if abs(surf.VKnot(i) - knot) < tol:
                            current_mult = surf.VMultiplicity(i)
                            break

                    if current_mult < mult:
                        surf.InsertVKnot(knot, mult - current_mult, tol, False)

        return result_surfaces

    def _create_gordon_surface(
        self,
        profiles: list[Geom_BSplineCurve],
        guides: list[Geom_BSplineCurve],
        intersection_params_spline_u: list[float],
        intersection_params_spline_v: list[float],
    ):
        """
        Create the Gordon surface using the C++ algorithm.

        This follows the exact same logic as the C++ implementation.
        """
        # print(f'profiles={len(profiles)}, guides={len(guides)}, intersection_params_spline_u={intersection_params_spline_u}, intersection_params_spline_v={intersection_params_spline_v}')

        # Check whether there are any u-directional and v-directional B-splines in the vectors
        if len(profiles) < 2:
            raise error(
                "There must be at least two profiles for the gordon surface.",
                ErrorCode.MATH_ERROR,
            )

        if len(guides) < 2:
            raise error(
                "There must be at least two guides for the gordon surface.",
                ErrorCode.MATH_ERROR,
            )

        # Check B-spline parametrization is equal among all curves
        umin = self.profiles[0].FirstParameter()
        umax = self.profiles[0].LastParameter()
        for profile in self.profiles:
            self._assert_range(profile, umin, umax, 1e-5)

        vmin = self.guides[0].FirstParameter()
        vmax = self.guides[0].LastParameter()
        for guide in self.guides:
            self._assert_range(guide, vmin, vmax, 1e-5)

        # Check curve network compatibility
        self._check_curve_network_compatibility(
            profiles,
            guides,
            intersection_params_spline_u,
            intersection_params_spline_v,
            self.tolerance,
        )

        # Create intersection points array
        n_u_params = len(intersection_params_spline_u)
        n_v_params = len(intersection_params_spline_v)
        intersection_points = TColgp_Array2OfPnt(1, n_u_params, 1, n_v_params)

        # Use splines in u-direction to get intersection points
        for spline_idx in range(len(profiles)):
            for intersection_idx in range(n_u_params):
                spline_u = profiles[spline_idx]
                parameter = intersection_params_spline_u[intersection_idx]
                intersection_points.SetValue(
                    intersection_idx + 1, spline_idx + 1, spline_u.Value(parameter)
                )

        # Check whether to build a closed continuous surface
        curve_u_tolerance = BSplineAlgorithms.REL_TOL_CLOSED * BSplineAlgorithms.scale(
            guides
        )
        curve_v_tolerance = BSplineAlgorithms.REL_TOL_CLOSED * BSplineAlgorithms.scale(
            profiles
        )
        tp_tolerance = BSplineAlgorithms.REL_TOL_CLOSED * BSplineAlgorithms.scale(
            intersection_points
        )

        make_u_closed = BSplineAlgorithms.is_u_dir_closed(
            intersection_points, tp_tolerance
        ) and guides[0].IsEqual(guides[-1], curve_u_tolerance)
        make_v_closed = BSplineAlgorithms.is_v_dir_closed(
            intersection_points, tp_tolerance
        ) and profiles[0].IsEqual(profiles[-1], curve_v_tolerance)

        # Skinning in v-direction with u directional B-Splines; u, v is normal
        surf_profiles_skinner = CurvesToSurface(
            list(profiles), intersection_params_spline_v, make_v_closed
        )
        surf_profiles = surf_profiles_skinner.surface()

        # Skinning in u-direction with v directional B-Splines; u, v is swapped
        surf_guides_skinner = CurvesToSurface(
            list(guides), intersection_params_spline_u, make_u_closed
        )
        surf_guides = surf_guides_skinner.surface()

        # flipping of the surface in v-direction;
        surf_guides = BSplineAlgorithms.flip_surface(surf_guides)

        # if there are too little points for degree in u-direction = 3 and degree in v-direction=3 creating an interpolation B-spline surface isn't possible in Open CASCADE
        # Open CASCADE doesn't have a B-spline surface interpolation method where one can give the u- and v-directional parameters as arguments
        tensor_prod_surf = GordonSurfaceBuilder._points_to_surface_internal(
            intersection_points,
            intersection_params_spline_u,
            intersection_params_spline_v,
            make_u_closed,
            make_v_closed,
        )

        # Match degree of all three surfaces
        degree_u = max(
            surf_guides.UDegree(), surf_profiles.UDegree(), tensor_prod_surf.UDegree()
        )

        degree_v = max(
            surf_guides.VDegree(), surf_profiles.VDegree(), tensor_prod_surf.VDegree()
        )

        # Elevate degrees to match maximum
        surf_guides.IncreaseDegree(degree_u, degree_v)
        surf_profiles.IncreaseDegree(degree_u, degree_v)
        tensor_prod_surf.IncreaseDegree(degree_u, degree_v)

        # create common knot vector for all three surfaces
        surfaces_vector = (
            GordonSurfaceBuilder._create_common_knots_vector_surface_internal(
                [surf_guides, surf_profiles, tensor_prod_surf], SurfaceDirection.both
            )
        )

        assert len(surfaces_vector) == 3

        self._surface_guides = surfaces_vector[0]
        self._surface_profiles = surfaces_vector[1]
        self._surface_intersections = surfaces_vector[2]

        # Verify surface dimensions match
        assert (
            self._surface_guides.NbUPoles() == self._surface_profiles.NbUPoles()
            and self._surface_profiles.NbUPoles()
            == self._surface_intersections.NbUPoles()
        )
        assert (
            self._surface_guides.NbVPoles() == self._surface_profiles.NbVPoles()
            and self._surface_profiles.NbVPoles()
            == self._surface_intersections.NbVPoles()
        )

        # Create the Gordon surface by manually copying the profile surface
        self._surface_gordon = clone_bspline_surface(self._surface_profiles)

        # Apply Gordon formula: S_Gordon = S_Profiles + S_Guides - S_TensorProduct
        for cp_u_idx in range(1, self._surface_gordon.NbUPoles() + 1):
            for cp_v_idx in range(1, self._surface_gordon.NbVPoles() + 1):
                cp_surf_u = self._surface_profiles.Pole(cp_u_idx, cp_v_idx)
                cp_surf_v = self._surface_guides.Pole(cp_u_idx, cp_v_idx)
                cp_tensor = self._surface_intersections.Pole(cp_u_idx, cp_v_idx)

                # S_Gordon = S_Profiles + S_Guides - S_TensorProduct
                new_pole_xyz = cp_surf_u.XYZ() + cp_surf_v.XYZ() - cp_tensor.XYZ()
                self._surface_gordon.SetPole(
                    cp_u_idx,
                    cp_v_idx,
                    gp_Pnt(new_pole_xyz.X(), new_pole_xyz.Y(), new_pole_xyz.Z()),
                )

    def _assert_range(self, curve, umin, umax, tol=1e-7):
        """Check if curve parameters are within expected range."""
        if (
            abs(curve.FirstParameter() - umin) > tol
            or abs(curve.LastParameter() - umax) > tol
        ):
            raise error(f"Curve not in range [{umin}, {umax}].", ErrorCode.MATH_ERROR)

    def _check_curve_network_compatibility(
        self,
        profiles: list[Geom_BSplineCurve],
        guides: list[Geom_BSplineCurve],
        intersection_params_spline_u,
        intersection_params_spline_v,
        tol,
    ):
        """Check if the curve network is compatible."""
        # Find the 'average' scale of the B-splines
        splines_scale = 0.5 * (
            BSplineAlgorithms.scale(profiles) + BSplineAlgorithms.scale(guides)
        )

        if (
            abs(intersection_params_spline_u[0]) > splines_scale * tol
            or abs(intersection_params_spline_u[-1] - 1.0) > splines_scale * tol
        ):
            raise error(
                "WARNING: B-splines in u-direction mustn't stick out, spline network must be 'closed'!",
                ErrorCode.MATH_ERROR,
            )

        if (
            abs(intersection_params_spline_v[0]) > splines_scale * tol
            or abs(intersection_params_spline_v[-1] - 1.0) > splines_scale * tol
        ):
            raise error(
                "WARNING: B-splines in v-direction mustn't stick out, spline network must be 'closed'!",
                ErrorCode.MATH_ERROR,
            )

        # Check compatibility of network
        for u_param_idx in range(len(intersection_params_spline_u)):
            spline_u_param = intersection_params_spline_u[u_param_idx]
            spline_v = guides[u_param_idx]

            for v_param_idx in range(len(intersection_params_spline_v)):
                spline_u = profiles[v_param_idx]
                spline_v_param = intersection_params_spline_v[v_param_idx]

                p_prof = spline_u.Value(spline_u_param)
                p_guid = spline_v.Value(spline_v_param)
                distance = p_prof.Distance(p_guid)

                # print(f'u_param_idx={u_param_idx}, v_param_idx={v_param_idx}, distance={distance}, splines_scale={splines_scale}, tol={tol}')
                if distance > splines_scale * tol:
                    print(
                        f"u_param_idx={u_param_idx}, v_param_idx={v_param_idx}, distance={distance}, splines_scale={splines_scale}, tol={tol}"
                    )
                    raise error(
                        "B-spline network is incompatible (e.g. wrong parametrization) or intersection parameters are in a wrong order!",
                        ErrorCode.MATH_ERROR,
                    )
