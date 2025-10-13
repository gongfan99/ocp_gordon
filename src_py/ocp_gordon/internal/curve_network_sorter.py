"""
Curve network sorting and ordering.

This module provides functionality to sort and order curves in a network
based on their intersection parameters to ensure consistent orientation.
"""

"""
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2017 German Aerospace Center (DLR)

Created: 2017 Martin Siggel <Martin.Siggel@dlr.de>
"""

import math
from typing import List, Tuple

import numpy as np
from OCP.Geom import Geom_BSplineCurve, Geom_Curve
from OCP.gp import gp_Pnt

from .error import error


class CurveNetworkSorter:
    """
    Sorts and orders curves in a network based on intersection parameters.
    """

    # returns the column index if the maximum of i-th row
    def max_row_index(self, m: np.ndarray, irow: int) -> int:
        max_val = float("-inf")
        jmax = -1  # Initialize to -1 to indicate no valid index found

        for jcol in range(m.shape[1]):
            # Skip zero-length guides
            if _is_zero_length_curve(self._guides[jcol]):
                continue

            # If this is the first valid guide found, set jmax to it
            if jmax == -1:
                max_val = m[irow, jcol]
                jmax = jcol
            elif m[irow, jcol] > max_val:
                max_val = m[irow, jcol]
                jmax = jcol
        return jmax

    # returns the row index if the maximum of i-th col
    def max_col_index(self, m: np.ndarray, jcol: int) -> int:
        max_val = float("-inf")
        imax = -1  # Initialize to -1 to indicate no valid index found

        for irow in range(m.shape[0]):
            # Skip zero-length profiles
            if _is_zero_length_curve(self._profiles[irow]):
                continue

            # If this is the first valid profile found, set imax to it
            if imax == -1:
                max_val = m[irow, jcol]
                imax = irow
            elif m[irow, jcol] > max_val:
                max_val = m[irow, jcol]
                imax = irow
        return imax

    # returns the column index if the minimum of i-th row
    def min_row_index(self, m: np.ndarray, irow: int) -> int:
        min_val = float("inf")
        jmin = -1  # Initialize to -1 to indicate no valid index found

        for jcol in range(m.shape[1]):
            # Skip zero-length guides
            if _is_zero_length_curve(self._guides[jcol]):
                continue

            # If this is the first valid guide found, set jmin to it
            if jmin == -1:
                min_val = m[irow, jcol]
                jmin = jcol
            elif m[irow, jcol] < min_val:
                min_val = m[irow, jcol]
                jmin = jcol
        return jmin

    # returns the row index if the minimum of i-th col
    def min_col_index(self, m: np.ndarray, jcol: int) -> int:
        min_val = float("inf")
        imin = -1  # Initialize to -1 to indicate no valid index found

        for irow in range(m.shape[0]):
            # Skip zero-length profiles
            if _is_zero_length_curve(self._profiles[irow]):
                continue

            # If this is the first valid profile found, set imin to it
            if imin == -1:
                min_val = m[irow, jcol]
                imin = irow
            elif m[irow, jcol] < min_val:
                min_val = m[irow, jcol]
                imin = irow
        return imin

    def __init__(
        self,
        profiles: list[Geom_Curve],
        guides: list[Geom_Curve],
        parms_inters_profiles: np.ndarray,
        parms_inters_guides: np.ndarray,
    ):

        self._profiles = profiles
        self._guides = guides
        self._parms_inters_profiles = parms_inters_profiles
        self._parms_inters_guides = parms_inters_guides
        self._has_performed = False

        # check consistency of input data
        n_profiles = len(profiles)
        n_guides = len(guides)

        if n_profiles != self._parms_inters_profiles.shape[0]:
            raise error("Invalid row size of parms_inters_profiles matrix.")

        if n_profiles != self._parms_inters_guides.shape[0]:
            raise error("Invalid row size of parms_inters_guides matrix.")

        if n_guides != self._parms_inters_profiles.shape[1]:
            raise error("Invalid col size of parms_inters_profiles matrix.")

        if n_guides != self._parms_inters_guides.shape[1]:
            raise error("Invalid col size of parms_inters_guides matrix.")

        # create helper vectors with indices
        self._prof_idx = [str(i) for i in range(n_profiles)]
        self._guid_idx = [str(i) for i in range(n_guides)]

    def perform(self):
        if self._has_performed:
            return

        prof_start = 0
        guide_start = 0

        guide_must_be_reversed = False
        prof_start, guide_start, guide_must_be_reversed = self.get_start_curve_indices()

        # put start curves first in array
        self.swap_profiles(0, prof_start)
        self.swap_guides(0, guide_start)

        if guide_must_be_reversed:
            self.reverse_guide(0)

        n_guides = self.NGuides()
        n_profiles = self.NProfiles()

        # Use the first non-zero-length profile to sort the guides
        # (self._profiles[0] is already guaranteed to be non-zero-length by get_start_curve_indices)
        for n in range(n_guides, 1, -1):
            for j in range(0, n - 1):
                if (
                    self._parms_inters_profiles[0, j]
                    > self._parms_inters_profiles[0, j + 1]
                ):
                    self.swap_guides(j, j + 1)

        # Use the first non-zero-length guide (after sorting) to sort the profiles
        first_non_zero_guide_idx = _find_first_non_zero_length_index(self._guides)
        if first_non_zero_guide_idx == -1:
            raise error("No non-zero-length guide found to sort profiles.")

        for n in range(n_profiles, 1, -1):
            for i in range(0, n - 1):
                if (
                    self._parms_inters_guides[i, first_non_zero_guide_idx]
                    > self._parms_inters_guides[i + 1, first_non_zero_guide_idx]
                ):
                    self.swap_profiles(i, i + 1)

        # reverse profiles, if necessary
        for i_prof in range(1, n_profiles):
            if (
                self._parms_inters_profiles[i_prof, 0]
                > self._parms_inters_profiles[i_prof, n_guides - 1]
            ):
                self.reverse_profile(i_prof)

        # reverse guide, if necessary
        for i_guid in range(1, n_guides):
            if (
                self._parms_inters_guides[0, i_guid]
                > self._parms_inters_guides[n_profiles - 1, i_guid]
            ):
                self.reverse_guide(i_guid)

        self._has_performed = True

    def swap_profiles(self, idx1: int, idx2: int):
        if idx1 == idx2:
            return

        self._profiles[idx1], self._profiles[idx2] = (
            self._profiles[idx2],
            self._profiles[idx1],
        )
        self._prof_idx[idx1], self._prof_idx[idx2] = (
            self._prof_idx[idx2],
            self._prof_idx[idx1],
        )

        # Swap rows in numpy arrays
        self._parms_inters_guides[[idx1, idx2], :] = self._parms_inters_guides[
            [idx2, idx1], :
        ]
        self._parms_inters_profiles[[idx1, idx2], :] = self._parms_inters_profiles[
            [idx2, idx1], :
        ]

    def swap_guides(self, idx1: int, idx2: int):
        if idx1 == idx2:
            return

        self._guides[idx1], self._guides[idx2] = self._guides[idx2], self._guides[idx1]
        self._guid_idx[idx1], self._guid_idx[idx2] = (
            self._guid_idx[idx2],
            self._guid_idx[idx1],
        )

        # Swap columns in numpy arrays
        self._parms_inters_guides[:, [idx1, idx2]] = self._parms_inters_guides[
            :, [idx2, idx1]
        ]
        self._parms_inters_profiles[:, [idx1, idx2]] = self._parms_inters_profiles[
            :, [idx2, idx1]
        ]

    def get_start_curve_indices(self) -> tuple[int, int, bool]:
        # find curves, that begin at the same point (have the smallest parameter at their intersection)
        # Prioritize non-zero-length curves
        for irow in range(self.NProfiles()):
            # Ensure the current profile is not zero-length
            if _is_zero_length_curve(self._profiles[irow]):
                continue

            jmin = self.min_row_index(self._parms_inters_profiles, irow)
            if jmin == -1:  # No non-zero-length guide found for this profile
                continue

            # The guide at jmin is guaranteed to be non-zero-length by min_row_index
            imin = self.min_col_index(self._parms_inters_guides, jmin)
            if imin == -1:  # No non-zero-length profile found for this guide
                continue

            # The profile at imin is guaranteed to be non-zero-length by min_col_index
            if imin == irow:
                # we found the start curves
                return imin, jmin, False

        # there are situations (a loop) when the previous situation does not exist
        # find curves where the start of a profile hits the end of a guide
        # Prioritize non-zero-length curves
        for irow in range(self.NProfiles()):
            # Ensure the current profile is not zero-length
            if _is_zero_length_curve(self._profiles[irow]):
                continue

            jmin = self.min_row_index(self._parms_inters_profiles, irow)
            if jmin == -1:  # No non-zero-length guide found for this profile
                continue

            # The guide at jmin is guaranteed to be non-zero-length by min_row_index
            imax = self.max_col_index(self._parms_inters_guides, jmin)
            if imax == -1:  # No non-zero-length profile found for this guide
                continue

            # The profile at imax is guaranteed to be non-zero-length by max_col_index
            if imax == irow:
                # we found the start curves
                return imax, jmin, True

        # we have not found the starting curve. The network seems invalid
        raise error("Cannot find starting curves of curve network.")

    def NProfiles(self) -> int:
        return len(self._profiles)

    def NGuides(self) -> int:
        return len(self._guides)

    def ProfileIndices(self) -> list[str]:
        return self._prof_idx

    def GuideIndices(self) -> list[str]:
        return self._guid_idx

    def reverse_profile(self, profile_idx: int):
        profile = self._profiles[profile_idx]

        last_parm = (
            profile.LastParameter()
            if profile
            else self._parms_inters_profiles[
                profile_idx,
                self.max_row_index(self._parms_inters_profiles, profile_idx),
            ]
        )
        first_parm = (
            profile.FirstParameter()
            if profile
            else self._parms_inters_profiles[
                profile_idx,
                self.min_row_index(self._parms_inters_profiles, profile_idx),
            ]
        )

        # compute new parameters
        for icol in range(self.NGuides()):
            self._parms_inters_profiles[profile_idx, icol] = (
                -self._parms_inters_profiles[profile_idx, icol] + first_parm + last_parm
            )

        if profile:
            profile.Reverse()

        self._prof_idx[profile_idx] = "-" + self._prof_idx[profile_idx]

    def reverse_guide(self, guide_idx: int):
        guide = self._guides[guide_idx]

        last_parm = (
            guide.LastParameter()
            if guide
            else self._parms_inters_guides[
                self.max_col_index(self._parms_inters_guides, guide_idx), guide_idx
            ]
        )
        first_parm = (
            guide.FirstParameter()
            if guide
            else self._parms_inters_guides[
                self.min_col_index(self._parms_inters_guides, guide_idx), guide_idx
            ]
        )

        # compute new parameter
        for irow in range(self.NProfiles()):
            self._parms_inters_guides[irow, guide_idx] = (
                -self._parms_inters_guides[irow, guide_idx] + first_parm + last_parm
            )

        if guide:
            guide.Reverse()

        self._guid_idx[guide_idx] = "-" + self._guid_idx[guide_idx]


def _is_zero_length_curve(curve: Geom_Curve, tol=1e-9) -> bool:
    if not isinstance(curve, Geom_BSplineCurve):
        return False  # Only BSplineCurves can be zero-length points in this context

    poles = [curve.Pole(i + 1) for i in range(curve.NbPoles())]
    if not poles:
        return True  # A curve with no poles can be considered zero-length
    ref = poles[0]
    for p in poles[1:]:
        if ref.Distance(p) > tol:
            return False
    return True


def _find_first_non_zero_length_index(curves: list[Geom_Curve]) -> int:
    for i, curve in enumerate(curves):
        # Only check for zero-length if it's a BSplineCurve, otherwise assume it's not a point
        if isinstance(curve, Geom_BSplineCurve):
            if not _is_zero_length_curve(curve):
                return i
        else:
            # For other Geom_Curve types, we assume they are not zero-length points
            # unless a more specific check is needed.
            return i
    return -1  # No non-zero-length curve found
