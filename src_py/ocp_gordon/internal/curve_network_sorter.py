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

import numpy as np
from OCP.Geom import Geom_Curve
from typing import List, Tuple
from .error import error
import math

# returns the column index if the maximum of i-th row
def max_row_index(m: np.ndarray, irow: int) -> int:
    max_val = float('-inf')
    jmax = 0

    for jcol in range(m.shape[1]):
        if m[irow, jcol] > max_val:
            max_val = m[irow, jcol]
            jmax = jcol
    return jmax

# returns the row index if the maximum of i-th col
def max_col_index(m: np.ndarray, jcol: int) -> int:
    max_val = float('-inf')
    imax = 0

    for irow in range(m.shape[0]):
        if m[irow, jcol] > max_val:
            max_val = m[irow, jcol]
            imax = irow
    return imax

# returns the column index if the minimum of i-th row
def min_row_index(m: np.ndarray, irow: int) -> int:
    min_val = float('inf')
    jmin = 0

    for jcol in range(m.shape[1]):
        if m[irow, jcol] < min_val:
            min_val = m[irow, jcol]
            jmin = jcol
    return jmin

# returns the row index if the minimum of i-th col
def min_col_index(m: np.ndarray, jcol: int) -> int:
    min_val = float('inf')
    imin = 0

    for irow in range(m.shape[0]):
        if m[irow, jcol] < min_val:
            min_val = m[irow, jcol]
            imin = irow
    return imin


class CurveNetworkSorter:
    """
    Sorts and orders curves in a network based on intersection parameters.
    """
    
    def __init__(self,
                 profiles: List[Geom_Curve],
                 guides: List[Geom_Curve],
                 parms_inters_profiles: np.ndarray,
                 parms_inters_guides: np.ndarray):
        
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

        # perform a bubble sort for the guides,
        # such that the guides intersection of the first profile are ascending
        for n in range(n_guides, 1, -1):
            for j in range(1, n - 1):
                if self._parms_inters_profiles[0, j] > self._parms_inters_profiles[0, j + 1]:
                    self.swap_guides(j, j + 1)

        # perform a bubble sort of the profiles,
        # such that the profiles are in ascending order of the first guide
        for n in range(n_profiles, 1, -1):
            for i in range(1, n - 1):
                if self._parms_inters_guides[i, 0] > self._parms_inters_guides[i + 1, 0]:
                    self.swap_profiles(i, i + 1)

        # reverse profiles, if necessary
        for i_prof in range(1, n_profiles):
            if self._parms_inters_profiles[i_prof, 0] > self._parms_inters_profiles[i_prof, n_guides - 1]:
                self.reverse_profile(i_prof)

        # reverse guide, if necessary
        for i_guid in range(1, n_guides):
            if self._parms_inters_guides[0, i_guid] > self._parms_inters_guides[n_profiles - 1, i_guid]:
                self.reverse_guide(i_guid)

        self._has_performed = True
    
    def swap_profiles(self, idx1: int, idx2: int):
        if idx1 == idx2:
            return

        self._profiles[idx1], self._profiles[idx2] = self._profiles[idx2], self._profiles[idx1]
        self._prof_idx[idx1], self._prof_idx[idx2] = self._prof_idx[idx2], self._prof_idx[idx1]
        
        # Swap rows in numpy arrays
        self._parms_inters_guides[[idx1, idx2], :] = self._parms_inters_guides[[idx2, idx1], :]
        self._parms_inters_profiles[[idx1, idx2], :] = self._parms_inters_profiles[[idx2, idx1], :]

    def swap_guides(self, idx1: int, idx2: int):
        if idx1 == idx2:
            return

        self._guides[idx1], self._guides[idx2] = self._guides[idx2], self._guides[idx1]
        self._guid_idx[idx1], self._guid_idx[idx2] = self._guid_idx[idx2], self._guid_idx[idx1]
        
        # Swap columns in numpy arrays
        self._parms_inters_guides[:, [idx1, idx2]] = self._parms_inters_guides[:, [idx2, idx1]]
        self._parms_inters_profiles[:, [idx1, idx2]] = self._parms_inters_profiles[:, [idx2, idx1]]

    def get_start_curve_indices(self) -> Tuple[int, int, bool]:
        # find curves, that begin at the same point (have the smallest parameter at their intersection)
        for irow in range(self.NProfiles()):
            jmin = min_row_index(self._parms_inters_profiles, irow)
            imin = min_col_index(self._parms_inters_guides, jmin)

            if imin == irow:
                # we found the start curves
                # print(f'irow={irow}, imin={imin}, jmin={jmin}')
                return imin, jmin, False

        # there are situations (a loop) when the previous situation does not exist
        # find curves where the start of a profile hits the end of a guide
        for irow in range(self.NProfiles()):
            jmin = min_row_index(self._parms_inters_profiles, irow)
            imax = max_col_index(self._parms_inters_guides, jmin)

            if imax == irow:
                # we found the start curves
                return imax, jmin, True

        # we have not found the starting curve. The network seems invalid
        raise error("Cannot find starting curves of curve network.")

    def NProfiles(self) -> int:
        return len(self._profiles)

    def NGuides(self) -> int:
        return len(self._guides)

    def ProfileIndices(self) -> List[str]:
        return self._prof_idx

    def GuideIndices(self) -> List[str]:
        return self._guid_idx

    def reverse_profile(self, profile_idx: int):
        profile = self._profiles[profile_idx]
        
        last_parm = profile.LastParameter() if profile else self._parms_inters_profiles[profile_idx, max_row_index(self._parms_inters_profiles, profile_idx)]
        first_parm = profile.FirstParameter() if profile else self._parms_inters_profiles[profile_idx, min_row_index(self._parms_inters_profiles, profile_idx)]

        # compute new parameters
        for icol in range(self.NGuides()):
            self._parms_inters_profiles[profile_idx, icol] = -self._parms_inters_profiles[profile_idx, icol] + first_parm + last_parm

        if profile:
            profile.Reverse()

        self._prof_idx[profile_idx] = "-" + self._prof_idx[profile_idx]

    def reverse_guide(self, guide_idx: int):
        guide = self._guides[guide_idx]
        
        last_parm = guide.LastParameter() if guide else self._parms_inters_guides[max_col_index(self._parms_inters_guides, guide_idx), guide_idx]
        first_parm = guide.FirstParameter() if guide else self._parms_inters_guides[min_col_index(self._parms_inters_guides, guide_idx), guide_idx]

        # compute new parameter
        for irow in range(self.NProfiles()):
            self._parms_inters_guides[irow, guide_idx] = -self._parms_inters_guides[irow, guide_idx] + first_parm + last_parm

        if guide:
            guide.Reverse()

        self._guid_idx[guide_idx] = "-" + self._guid_idx[guide_idx]
