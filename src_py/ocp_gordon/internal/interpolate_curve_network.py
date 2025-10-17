"""
Main interface for Gordon curve interpolation.

This module provides the main interpolate_curve_network function that serves
as the entry point for the Gordon surface interpolation algorithm.
"""

from typing import List, Optional, Union

import numpy as np
from OCP.Geom import Geom_BSplineCurve, Geom_BSplineSurface, Geom_Curve
from OCP.Precision import Precision

from .bspline_algorithms import BSplineAlgorithms
from .curve_network_sorter import (
    CurveNetworkSorter,
    _find_first_non_zero_length_index,
    _is_zero_length_curve,
)
from .error import (  # Ensure error and ErrorCode are imported for internal use
    ErrorCode,
    error,
)
from .gordon_surface_builder import GordonSurfaceBuilder
from .intersect_bsplines import IntersectBSplines
from .misc import save_bsplines_to_file


class GordonInterpolationError(Exception):
    """Base exception for Gordon interpolation errors."""

    pass


class InvalidInputError(GordonInterpolationError):
    """Exception raised for invalid input parameters."""

    pass


class IntersectionError(GordonInterpolationError):
    """Exception raised for intersection-related errors."""

    pass


class CompatibilityError(GordonInterpolationError):
    """Exception raised for curve compatibility issues."""

    pass


class SurfaceConstructionError(GordonInterpolationError):
    """Exception raised for surface construction failures."""

    pass


def interpolate_curve_network(
    ucurves: list[Geom_Curve | Geom_BSplineCurve],
    vcurves: list[Geom_Curve | Geom_BSplineCurve],
    tolerance: float = 3e-4,
) -> Geom_BSplineSurface:
    """
    Interpolates the curve network by a B-spline surface using Gordon method.

    The u curves and v curves must intersect each other within the tolerance.

    Note: The input curves are reparametrized to fulfill the compatibility
    criteria of the Gordon method, which might introduce a small error.

    Args:
        ucurves: Multiple curves that will be interpolated in u direction
        vcurves: Multiple curves that will be interpolated in v direction,
                 must intersect the ucurves
        tolerance: Tolerance in which the u- and v-curves need to intersect

    Returns:
        Geom_BSplineSurface: The interpolated Gordon surface

    Raises:
        error: For various interpolation errors with specific ErrorCode.
    """
    # Convert generic curves to B-splines if needed
    u_bsplines = BSplineAlgorithms.to_bsplines(ucurves)
    v_bsplines = BSplineAlgorithms.to_bsplines(vcurves)

    # Create the interpolator
    interpolator = InterpolateCurveNetwork(u_bsplines, v_bsplines, tolerance)

    # Perform the interpolation
    return interpolator.surface()


class InterpolateCurveNetwork:
    """
    Curve network interpolation with Gordon surfaces.

    The algorithm uses the Gordon method to create the interpolation surface.
    It performs the following steps:
    - Compute intersection points between profiles and guides
    - Sort the profiles and guides
    - Reparametrize profiles and curves to make the network compatible
    - Compute the Gordon surface
    """

    def __init__(
        self,
        profiles: list[Geom_BSplineCurve],
        guides: list[Geom_BSplineCurve],
        spatial_tolerance: float,
    ):
        """
        Initialize the curve network interpolator.

        Args:
            profiles: The profiles to be interpolated
            guides: The guide curves to be interpolated
            spatial_tolerance: Maximum allowed distance between each guide and profile
        """
        # C++ constructor validation and duplicate removal
        if len(profiles) < 2:
            raise InvalidInputError(
                "There must be at least two profiles for the curve network interpolation."
            )
        if len(guides) < 2:
            raise InvalidInputError(
                "There must be at least two guides for the curve network interpolation."
            )

        # Remove duplicates from profiles and guides (integrated logic)
        unique_profiles: list[Geom_BSplineCurve] = []
        for profile in profiles:
            is_unique = True
            for unique_profile in unique_profiles:
                if profile.IsEqual(unique_profile, Precision.PConfusion_s()):
                    is_unique = False
                    break
            if is_unique:
                unique_profiles.append(profile)

        unique_guides: list[Geom_BSplineCurve] = []
        for guide in guides:
            is_unique = True
            for unique_guide in unique_guides:
                if guide.IsEqual(unique_guide, Precision.PConfusion_s()):
                    is_unique = False
                    break
            if is_unique:
                unique_guides.append(guide)

        if len(unique_profiles) < 2:
            raise InvalidInputError(
                "There must be at least two unique profiles for the curve network interpolation."
            )
        if len(unique_guides) < 2:
            raise InvalidInputError(
                "There must be at least two unique guides for the curve network interpolation."
            )

        self.profiles = unique_profiles
        self.guides = unique_guides

        self.spatial_tol = spatial_tolerance
        self.has_performed = False

        # Results storage
        self.skinning_surf_profiles: Geom_BSplineSurface | None = None
        self.skinning_surf_guides: Geom_BSplineSurface | None = None
        self.tensor_prod_surf: Geom_BSplineSurface | None = None
        self.gordon_surf: Geom_BSplineSurface | None = None
        self.intersection_params_u: list[float] = (
            []
        )  # Stores final reparametrized parameters
        self.intersection_params_v: list[float] = (
            []
        )  # Stores final reparametrized parameters

    def perform(self):
        """Perform the complete Gordon surface interpolation."""
        if self.has_performed:
            return

        # Gordon surfaces are only defined on a compatible curve network
        # We first have to reparametrize the network
        self._make_curves_compatible()

        # Build Gordon surface
        # It needs the final intersection parameters and reparametrized curves.
        # The reparametrized curves are already in self.profiles and self.guides.
        # The final intersection parameters are stored in self.intersection_params_u/v
        # after _make_curves_compatible.

        builder = GordonSurfaceBuilder(
            self.profiles,
            self.guides,
            self.intersection_params_u,  # These should be the final reparametrized parameters
            self.intersection_params_v,  # These should be the final reparametrized parameters
            self.spatial_tol,
        )

        self.gordon_surf = builder.surface_gordon()
        self.skinning_surf_profiles = builder.surface_profiles()
        self.skinning_surf_guides = builder.surface_guides()
        self.tensor_prod_surf = builder.surface_intersections()

        # assert self.gordon_surf.Value(1, 0).IsEqual(self.profiles[0].Value(1.0), 1e-4)

        self._ensure_c2()  # Ensure C2 continuity

        self.has_performed = True

    def surface(self) -> Geom_BSplineSurface:
        """Returns the final interpolated Gordon surface."""
        self.perform()  # Ensure perform is called if not already
        assert self.gordon_surf is not None
        return self.gordon_surf

    def surface_profiles(self) -> Geom_BSplineSurface:
        """Returns the surface that interpolates the profiles."""
        self.perform()
        assert self.skinning_surf_profiles is not None
        return self.skinning_surf_profiles

    def surface_guides(self) -> Geom_BSplineSurface:
        """Returns the surface that interpolates the guides."""
        self.perform()
        assert self.skinning_surf_guides is not None
        return self.skinning_surf_guides

    def surface_intersections(self) -> Geom_BSplineSurface:
        """Returns the surface that interpolates the intersection points."""
        self.perform()
        assert self.tensor_prod_surf is not None
        return self.tensor_prod_surf

    def parameters_profiles(self) -> list[float]:
        """Returns the parameters at the profile curves (v-direction)."""
        self.perform()
        return self.intersection_params_u  # intersection_params_v in C++, likely error

    def parameters_guides(self) -> list[float]:
        """Returns the parameters at the guide curves (u-direction)."""
        self.perform()
        return self.intersection_params_v  # intersection_params_u in C++, likely error

    def _compute_intersections_matrix(
        self,
        initial_intersection_params_u: np.ndarray,
        initial_intersection_params_v: np.ndarray,
    ):
        """
        Computes the intersection parameters between profiles and guides.
        Mirrors the C++ ComputeIntersections method.
        """
        n_profiles = len(self.profiles)
        n_guides = len(self.guides)

        for spline_u_idx in range(n_profiles):
            for spline_v_idx in range(n_guides):
                # Use IntersectBSplines to find intersections
                intersections = IntersectBSplines(
                    self.profiles[spline_u_idx],
                    self.guides[spline_v_idx],
                    self.spatial_tol,
                )

                if not intersections:
                    # save_bsplines_to_file(
                    #     [self.profiles[spline_u_idx], self.guides[spline_v_idx]],
                    #     "curve.json",
                    # )
                    raise IntersectionError(
                        f"U-directional B-spline {spline_u_idx} and V-directional B-spline {spline_v_idx} don't intersect!"
                    )
                elif len(intersections) == 1:
                    initial_intersection_params_u[spline_u_idx, spline_v_idx] = (
                        intersections[0]["parmOnCurve1"]
                    )
                    initial_intersection_params_v[spline_u_idx, spline_v_idx] = (
                        intersections[0]["parmOnCurve2"]
                    )
                elif len(intersections) == 2:
                    # For closed curves, take the smaller parameter values
                    initial_intersection_params_u[spline_u_idx, spline_v_idx] = min(
                        intersections[0]["parmOnCurve1"],
                        intersections[1]["parmOnCurve1"],
                    )
                    initial_intersection_params_v[spline_u_idx, spline_v_idx] = min(
                        intersections[0]["parmOnCurve2"],
                        intersections[1]["parmOnCurve2"],
                    )
                else:  # len(intersections) > 2
                    raise IntersectionError(
                        f"U-directional B-spline {spline_u_idx} and V-directional B-spline {spline_v_idx} have more than two intersections! "
                        "Closed in both U and V directions surface isn't supported at this time."
                    )
        # print(initial_intersection_params_u)
        # print(initial_intersection_params_v)

    def _make_curves_compatible(self):
        """
        Make the curve network compatible by reparameterization and sorting.
        This method now encompasses steps previously in separate C++ methods.
        """
        # 1. Reparametrize into [0,1] range
        for profile in self.profiles:
            BSplineAlgorithms.reparametrize_bspline(profile, 0.0, 1.0, 1e-15)
        for guide in self.guides:
            BSplineAlgorithms.reparametrize_bspline(guide, 0.0, 1.0, 1e-15)

        # 2. Compute initial intersection parameters
        n_profiles = len(self.profiles)
        n_guides = len(self.guides)

        initial_intersection_params_u = np.zeros((n_profiles, n_guides))
        initial_intersection_params_v = np.zeros((n_profiles, n_guides))

        self._compute_intersections_matrix(
            initial_intersection_params_u, initial_intersection_params_v
        )

        # 3. Sort curves and update intersection parameters
        sorter = CurveNetworkSorter(
            list(self.profiles),
            list(self.guides),
            initial_intersection_params_u,
            initial_intersection_params_v,
        )
        sorter.perform()

        # Update curves and intersection parameters after sorting
        # Convert to List[Geom_BSplineCurve] as sorter._profiles/guides are List[Geom_Curve]
        self.profiles = BSplineAlgorithms.to_bsplines(sorter._profiles)
        self.guides = BSplineAlgorithms.to_bsplines(sorter._guides)

        # ensure zero-length curve only occurs at beginning and end
        for i in range(1, len(self.profiles) - 1):
            if _is_zero_length_curve(self.profiles[i]):
                raise ValueError(
                    f"Profile#{i} is a point. Points are only permitted at the beginning and end."
                )

        for i in range(1, len(self.guides) - 1):
            if _is_zero_length_curve(self.guides[i]):
                raise ValueError(
                    f"Guides#{i} is a point. Points are only permitted at the beginning and end."
                )

        # Correcting attribute access based on C++ naming convention (snake_case)
        # The CurveNetworkSorter class exposes intersection parameters as attributes
        sorted_intersection_params_u = sorter._parms_inters_profiles
        sorted_intersection_params_v = sorter._parms_inters_guides

        # 4. Handle closed curves (complex logic from C++)
        # Find the first non-zero-length profile and guide for closed curve checks
        first_non_zero_profile_idx = _find_first_non_zero_length_index(
            list(self.profiles)
        )
        first_non_zero_guide_idx = _find_first_non_zero_length_index(list(self.guides))

        if first_non_zero_profile_idx == -1:
            raise CompatibilityError(
                "No non-zero-length profile found for closed curve check."
            )
        if first_non_zero_guide_idx == -1:
            raise CompatibilityError(
                "No non-zero-length guide found for closed curve check."
            )

        first_non_zero_profile = self.profiles[first_non_zero_profile_idx]
        first_non_zero_guide = self.guides[first_non_zero_guide_idx]

        is_closed_profile = (
            first_non_zero_profile.IsClosed() or first_non_zero_profile.IsPeriodic()
        )
        is_closed_guides = (
            first_non_zero_guide.IsClosed() or first_non_zero_guide.IsPeriodic()
        )

        # C++ logic for handling closed curves and resizing matrices
        # It uses tmp_intersection_params_u/v (which are the sorted ones)
        # and populates intersection_params_u/v (newly sized matrices).

        # If both are closed, C++ enters the first 'if' block (closed profile),
        # duplicates the guide, and then the 'else if' for closed guides is skipped.
        # This implies only one direction can be closed at a time.
        if is_closed_profile and is_closed_guides:
            raise CompatibilityError(
                "Both profiles and guides cannot be closed simultaneously."
            )

        # Initialize final matrices with potentially adjusted dimensions
        current_n_profiles = len(self.profiles)
        current_n_guides = len(self.guides)

        final_intersection_params_u = np.zeros((current_n_profiles, current_n_guides))
        final_intersection_params_v = np.zeros((current_n_profiles, current_n_guides))

        if is_closed_profile:
            # Duplicate the first guide at the end
            self.guides.append(self.guides[0])
            current_n_guides += 1  # Update count

            # Resize final matrices to accommodate the new guide
            final_intersection_params_u = np.zeros(
                (current_n_profiles, current_n_guides)
            )
            final_intersection_params_v = np.zeros(
                (current_n_profiles, current_n_guides)
            )

            # Copy existing data from sorted matrices
            original_n_profiles = (
                current_n_profiles  # Number of profiles before potential duplication
            )
            original_n_guides = (
                current_n_guides - 1
            )  # Number of guides before duplication

            final_intersection_params_u[:original_n_profiles, :original_n_guides] = (
                sorted_intersection_params_u
            )
            final_intersection_params_v[:original_n_profiles, :original_n_guides] = (
                sorted_intersection_params_v
            )

            # Handle the duplicated guide (last column)
            for i in range(original_n_profiles):
                # C++: intersection_params_u(spline_u_idx, nGuides - 1) = tmp_intersection_params_u(spline_u_idx, 0) < BSplineAlgorithms::PAR_CHECK_TOL ? 1.0 : tmp_intersection_params_u(spline_u_idx, 0);
                # C++: intersection_params_v(spline_u_idx, nGuides - 1) = tmp_intersection_params_v(spline_u_idx, 0);

                val_u = sorted_intersection_params_u[i, 0]
                final_intersection_params_u[i, original_n_guides] = (
                    1.0 if abs(val_u) < BSplineAlgorithms.PAR_CHECK_TOL else val_u
                )
                final_intersection_params_v[i, original_n_guides] = (
                    sorted_intersection_params_v[i, 0]
                )

        elif is_closed_guides:
            # Duplicate the first profile at the end
            self.profiles.append(self.profiles[0])
            current_n_profiles += 1  # Update count

            # Resize final matrices to accommodate the new profile
            final_intersection_params_u = np.zeros(
                (current_n_profiles, current_n_guides)
            )
            final_intersection_params_v = np.zeros(
                (current_n_profiles, current_n_guides)
            )

            # Copy existing data from sorted matrices
            original_n_profiles = (
                current_n_profiles - 1
            )  # Number of profiles before duplication
            original_n_guides = (
                current_n_guides  # Number of guides before potential duplication
            )

            final_intersection_params_u[:original_n_profiles, :original_n_guides] = (
                sorted_intersection_params_u
            )
            final_intersection_params_v[:original_n_profiles, :original_n_guides] = (
                sorted_intersection_params_v
            )

            # Handle the duplicated profile (last row)
            for j in range(original_n_guides):
                # C++: intersection_params_u(nProfiles - 1, spline_v_idx) = tmp_intersection_params_u(0, spline_v_idx);
                # C++: intersection_params_v(nProfiles - 1, spline_v_idx) = tmp_intersection_params_v(0, spline_v_idx) < BSplineAlgorithms::PAR_CHECK_TOL ? 1.0 : tmp_intersection_params_v(0, spline_v_idx);

                val_v = sorted_intersection_params_v[0, j]
                final_intersection_params_u[original_n_profiles, j] = (
                    sorted_intersection_params_u[0, j]
                )
                final_intersection_params_v[original_n_profiles, j] = (
                    1.0 if abs(val_v) < BSplineAlgorithms.PAR_CHECK_TOL else val_v
                )
        else:
            # No closed curves, use the sorted matrices directly
            final_intersection_params_u = sorted_intersection_params_u
            final_intersection_params_v = sorted_intersection_params_v

        # 5. Eliminate inaccuracies in network intersections
        self._eliminate_inaccuracies_network_intersections(
            self.profiles,
            self.guides,
            final_intersection_params_u,
            final_intersection_params_v,
        )

        # 6. Compute new average parameters for reparameterization
        # Get current number of profiles and guides after potential duplication
        current_n_profiles = len(self.profiles)
        current_n_guides = len(self.guides)

        new_parameters_profiles = []
        for j in range(current_n_guides):  # Iterate over guides (columns)
            sum_u = 0.0
            count_non_zero_profiles = 0
            for i in range(current_n_profiles):  # Iterate over profiles (rows)
                if isinstance(
                    self.profiles[i], Geom_BSplineCurve
                ) and _is_zero_length_curve(self.profiles[i]):
                    continue  # Exclude zero-length profiles
                sum_u += final_intersection_params_u[i, j]
                count_non_zero_profiles += 1
            new_parameters_profiles.append(
                sum_u / count_non_zero_profiles if count_non_zero_profiles > 0 else 0.0
            )

        new_parameters_guides = []
        for i in range(current_n_profiles):  # Iterate over profiles (rows)
            sum_v = 0.0
            count_non_zero_guides = 0
            for j in range(current_n_guides):  # Iterate over guides (columns)
                if isinstance(
                    self.guides[j], Geom_BSplineCurve
                ) and _is_zero_length_curve(self.guides[j]):
                    continue  # Exclude zero-length guides
                sum_v += final_intersection_params_v[i, j]
                count_non_zero_guides += 1
            new_parameters_guides.append(
                sum_v / count_non_zero_guides if count_non_zero_guides > 0 else 0.0
            )

        # Validate that intersections are at the beginning
        if (
            new_parameters_profiles[0] > BSplineAlgorithms.PAR_CHECK_TOL
            or new_parameters_guides[0] > BSplineAlgorithms.PAR_CHECK_TOL
        ):
            raise CompatibilityError(
                "At least one B-spline has no intersection at the beginning."
            )

        # 7. Determine appropriate number of control points
        max_cp_u_orig = max((profile.NbPoles() for profile in self.profiles), default=0)
        max_cp_v_orig = max((guide.NbPoles() for guide in self.guides), default=0)

        min_cp = 10
        max_cp = 80

        min_u = max(current_n_guides + 2, min_cp)
        min_v = max(current_n_profiles + 2, min_cp)

        max_u = max(min_u, max_cp)
        max_v = max(min_v, max_cp)

        final_max_cp_u = self._clamp(max_cp_u_orig + 10, min_u, max_u)
        final_max_cp_v = self._clamp(max_cp_v_orig + 10, min_v, max_v)

        # 8. Reparametrize profiles and guides again using new parameters and control points
        # Reparametrize profiles
        for i in range(current_n_profiles):
            if isinstance(
                self.profiles[i], Geom_BSplineCurve
            ) and _is_zero_length_curve(self.profiles[i]):
                continue  # Skip zero-length profiles

            old_parameters = [
                final_intersection_params_u[i, j] for j in range(current_n_guides)
            ]

            # Eliminate small inaccuracies at the first and last knots
            if abs(old_parameters[0]) < BSplineAlgorithms.PAR_CHECK_TOL:
                old_parameters[0] = 0.0
            if abs(new_parameters_profiles[0]) < BSplineAlgorithms.PAR_CHECK_TOL:
                new_parameters_profiles[0] = 0.0
            if abs(old_parameters[-1] - 1.0) < BSplineAlgorithms.PAR_CHECK_TOL:
                old_parameters[-1] = 1.0
            if abs(new_parameters_profiles[-1] - 1.0) < BSplineAlgorithms.PAR_CHECK_TOL:
                new_parameters_profiles[-1] = 1.0

            # C++ uses reparametrizeBSplineContinuouslyApprox.
            result = BSplineAlgorithms.reparametrize_bspline_continuously_approx(
                self.profiles[i],
                old_parameters,
                new_parameters_profiles,
                final_max_cp_u,
            )
            if result.curve is not None:
                self.profiles[i] = result.curve

        # Reparametrize guides
        for j in range(current_n_guides):
            if isinstance(self.guides[j], Geom_BSplineCurve) and _is_zero_length_curve(
                self.guides[j]
            ):
                continue  # Skip zero-length guides

            old_parameters = [
                final_intersection_params_v[i, j] for i in range(current_n_profiles)
            ]

            # Eliminate small inaccuracies
            if abs(old_parameters[0]) < BSplineAlgorithms.PAR_CHECK_TOL:
                old_parameters[0] = 0.0
            if abs(new_parameters_guides[0]) < BSplineAlgorithms.PAR_CHECK_TOL:
                new_parameters_guides[0] = 0.0
            if abs(old_parameters[-1] - 1.0) < BSplineAlgorithms.PAR_CHECK_TOL:
                old_parameters[-1] = 1.0
            if abs(new_parameters_guides[-1] - 1.0) < BSplineAlgorithms.PAR_CHECK_TOL:
                new_parameters_guides[-1] = 1.0

            # C++ uses reparametrizeBSplineContinuouslyApprox.
            result = BSplineAlgorithms.reparametrize_bspline_continuously_approx(
                self.guides[j], old_parameters, new_parameters_guides, final_max_cp_v
            )
            if result.curve is not None:
                self.guides[j] = result.curve

        # Store the final intersection parameters (these are the new average parameters)
        self.intersection_params_u = new_parameters_profiles
        self.intersection_params_v = new_parameters_guides

    def _eliminate_inaccuracies_network_intersections(
        self,
        sorted_splines_u,
        sorted_splines_v,
        intersection_params_u,
        intersection_params_v,
    ):
        """Eliminate inaccuracies in network intersections."""
        n_profiles = len(sorted_splines_u)
        n_guides = len(sorted_splines_v)

        # eliminate small inaccuracies of the intersection parameters:
        # first intersection
        for spline_u_idx in range(n_profiles):
            if (
                abs(
                    intersection_params_u[spline_u_idx, 0] - sorted_splines_u[0].Knot(1)
                )
                < 0.001
            ):
                if abs(sorted_splines_u[0].Knot(1)) < 1e-10:
                    intersection_params_u[spline_u_idx, 0] = 0
                else:
                    intersection_params_u[spline_u_idx, 0] = sorted_splines_u[0].Knot(1)

        for spline_v_idx in range(n_guides):
            if (
                abs(
                    intersection_params_v[0, spline_v_idx] - sorted_splines_v[0].Knot(1)
                )
                < 0.001
            ):
                if abs(sorted_splines_v[0].Knot(1)) < 1e-10:
                    intersection_params_v[0, spline_v_idx] = 0
                else:
                    intersection_params_v[0, spline_v_idx] = sorted_splines_v[0].Knot(1)

        # last intersection
        for spline_u_idx in range(n_profiles):
            last_knot_u = sorted_splines_u[0].Knot(sorted_splines_u[0].NbKnots())
            if (
                abs(intersection_params_u[spline_u_idx, n_guides - 1] - last_knot_u)
                < 0.001
            ):
                intersection_params_u[spline_u_idx, n_guides - 1] = last_knot_u

        for spline_v_idx in range(n_guides):
            last_knot_v = sorted_splines_v[0].Knot(sorted_splines_v[0].NbKnots())
            if (
                abs(intersection_params_v[n_profiles - 1, spline_v_idx] - last_knot_v)
                < 0.001
            ):
                intersection_params_v[n_profiles - 1, spline_v_idx] = last_knot_v

    def _ensure_c2(self):
        """Ensure C2 continuity of the resulting surface."""
        if self.gordon_surf is None:
            return

        u_degree = self.gordon_surf.UDegree()
        v_degree = self.gordon_surf.VDegree()

        min_u_mult = max(1, u_degree - 2)
        min_v_mult = max(1, v_degree - 2)

        n_u_knots = self.gordon_surf.NbUKnots()
        for i in range(2, n_u_knots):  # Skip first and last knots
            if self.gordon_surf.UMultiplicity(i) > min_u_mult:
                self.gordon_surf.RemoveUKnot(i, min_u_mult, self.spatial_tol)

        n_v_knots = self.gordon_surf.NbVKnots()
        for i in range(2, n_v_knots):  # Skip first and last knots
            if self.gordon_surf.VMultiplicity(i) > min_v_mult:
                self.gordon_surf.RemoveVKnot(i, min_v_mult, self.spatial_tol)

    def _clamp(self, val, min_val, max_val):
        """Clamp a value between min and max."""
        if min_val > max_val:
            raise ValueError("Minimum may not be larger than maximum in clamp!")
        return max(min_val, min(val, max_val))


def interpolate_curve_network_debug(
    ucurves: list[Geom_Curve | Geom_BSplineCurve],
    vcurves: list[Geom_Curve | Geom_BSplineCurve],
    tolerance: float = 3e-4,
) -> InterpolateCurveNetwork:
    """
    Interpolates the curve network by a B-spline surface using Gordon method.

    The u curves and v curves must intersect each other within the tolerance.

    Note: The input curves are reparametrized to fulfill the compatibility
    criteria of the Gordon method, which might introduce a small error.

    Args:
        ucurves: Multiple curves that will be interpolated in u direction
        vcurves: Multiple curves that will be interpolated in v direction,
                 must intersect the ucurves
        tolerance: Tolerance in which the u- and v-curves need to intersect

    Returns:
        InterpolateCurveNetwork: The interpolater

    Raises:
        error: For various interpolation errors with specific ErrorCode.
    """
    # Convert generic curves to B-splines if needed
    u_bsplines = BSplineAlgorithms.to_bsplines(ucurves)
    v_bsplines = BSplineAlgorithms.to_bsplines(vcurves)

    # Create the interpolator
    interpolator = InterpolateCurveNetwork(u_bsplines, v_bsplines, tolerance)

    # Perform the interpolation
    return interpolator
