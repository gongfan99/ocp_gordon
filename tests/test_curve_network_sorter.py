import pytest
import sys
import os
import numpy as np
from typing import List, Tuple

from OCP.Geom import Geom_BSplineCurve
from OCP.gp import gp_Pnt
from typing import List, Tuple

import numpy as np

from OCP.Geom import Geom_BSplineCurve, Geom_Curve
from OCP.gp import gp_Pnt
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the CurveNetworkSorter from the internal module
from src_py.ocp_gordon.internal.curve_network_sorter import (
    CurveNetworkSorter,
    max_col_index,
    max_row_index,
    min_col_index,
    min_row_index,
)
from src_py.ocp_gordon.internal.error import error


def create_linear_bspline_curve(
    start_point: gp_Pnt, end_point: gp_Pnt, degree: int = 1
) -> Geom_BSplineCurve:
    """
    Helper to create a simple linear B-spline curve.
    """
    control_points = TColgp_Array1OfPnt(1, 2)
    control_points.SetValue(1, start_point)
    control_points.SetValue(2, end_point)

    knots = TColStd_Array1OfReal(1, 2)
    knots.SetValue(1, 0.0)
    knots.SetValue(2, 1.0)

    multiplicities = TColStd_Array1OfInteger(1, 2) # Corrected type
    multiplicities.SetValue(1, degree + 1)
    multiplicities.SetValue(2, degree + 1)

    curve = Geom_BSplineCurve(control_points, knots, multiplicities, degree)
    return curve


@pytest.fixture
def setup_sorter_data() -> tuple[
    list[Geom_BSplineCurve], list[Geom_BSplineCurve], np.ndarray, np.ndarray
]:
    # Create actual Geom_BSplineCurve instances
    profiles = [
        create_linear_bspline_curve(gp_Pnt(0, 0, 0), gp_Pnt(1, 0, 0)),
        create_linear_bspline_curve(gp_Pnt(0, 1, 0), gp_Pnt(1, 1, 0)),
        create_linear_bspline_curve(gp_Pnt(0, 2, 0), gp_Pnt(1, 2, 0)),
    ]
    guides = [
        create_linear_bspline_curve(gp_Pnt(0, 0, 0), gp_Pnt(0, 2, 0)),
        create_linear_bspline_curve(gp_Pnt(0.5, 0, 0), gp_Pnt(0.5, 2, 0)),
        create_linear_bspline_curve(gp_Pnt(1, 0, 0), gp_Pnt(1, 2, 0)),
    ]
    parms_inters_profiles = np.array(
        [[0.1, 0.5, 0.9], [0.2, 0.6, 0.8], [0.3, 0.7, 0.75]], dtype=float
    )
    parms_inters_guides = np.array(
        [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7], [0.9, 0.8, 0.75]], dtype=float
    )
    return profiles, guides, parms_inters_profiles, parms_inters_guides


def test_constructor_valid_input(setup_sorter_data):
    profiles, guides, parms_inters_profiles, parms_inters_guides = setup_sorter_data
    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(
        profiles=profiles,
        guides=guides,
        parms_inters_profiles=parms_inters_profiles,
        parms_inters_guides=parms_inters_guides,
    )
    assert sorter is not None
    assert sorter.NProfiles() == 3
    assert sorter.NGuides() == 3
    assert not sorter._has_performed


def test_constructor_invalid_row_size_profiles(setup_sorter_data):
    profiles, guides, _, parms_inters_guides = setup_sorter_data
    invalid_parms = np.array([[0.1, 0.5]], dtype=float)  # 1 row instead of 3
    with pytest.raises(error, match="Invalid row size of parms_inters_profiles matrix."):
        # Explicitly cast to List[Geom_Curve]
        CurveNetworkSorter(
            profiles=profiles,
            guides=guides,
            parms_inters_profiles=invalid_parms,
            parms_inters_guides=parms_inters_guides,
        )


def test_constructor_invalid_col_size_profiles(setup_sorter_data):
    profiles, guides, _, parms_inters_guides = setup_sorter_data
    invalid_parms = np.array([[0.1], [0.2], [0.3]], dtype=float)  # 1 col instead of 3
    with pytest.raises(error, match="Invalid col size of parms_inters_profiles matrix."):
        # Explicitly cast to List[Geom_Curve]
        CurveNetworkSorter(
            profiles=profiles,
            guides=guides,
            parms_inters_profiles=invalid_parms,
            parms_inters_guides=parms_inters_guides,
        )


def test_swap_profiles(setup_sorter_data):
    profiles, guides, parms_inters_profiles, parms_inters_guides = setup_sorter_data
    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(
        profiles=profiles,
        guides=guides,
        parms_inters_profiles=parms_inters_profiles.copy(),
        parms_inters_guides=parms_inters_guides.copy(),
    )

    original_profile_0 = sorter._profiles[0]
    original_profile_2 = sorter._profiles[2]
    original_prof_idx_0 = sorter._prof_idx[0]
    original_prof_idx_2 = sorter._prof_idx[2]
    original_parms_profiles_row_0 = sorter._parms_inters_profiles[0, :].copy()
    original_parms_profiles_row_2 = sorter._parms_inters_profiles[2, :].copy()
    original_parms_guides_row_0 = sorter._parms_inters_guides[0, :].copy()
    original_parms_guides_row_2 = sorter._parms_inters_guides[2, :].copy()

    sorter.swap_profiles(0, 2)

    assert sorter._profiles[0] == original_profile_2
    assert sorter._profiles[2] == original_profile_0
    assert sorter._prof_idx[0] == original_prof_idx_2
    assert sorter._prof_idx[2] == original_prof_idx_0
    np.testing.assert_array_equal(sorter._parms_inters_profiles[0, :], original_parms_profiles_row_2)
    np.testing.assert_array_equal(sorter._parms_inters_profiles[2, :], original_parms_profiles_row_0)
    np.testing.assert_array_equal(sorter._parms_inters_guides[0, :], original_parms_guides_row_2)
    np.testing.assert_array_equal(sorter._parms_inters_guides[2, :], original_parms_guides_row_0)


def test_swap_guides(setup_sorter_data):
    profiles, guides, parms_inters_profiles, parms_inters_guides = setup_sorter_data
    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(
        profiles=profiles,
        guides=guides,
        parms_inters_profiles=parms_inters_profiles.copy(),
        parms_inters_guides=parms_inters_guides.copy(),
    )

    original_guide_0 = sorter._guides[0]
    original_guide_2 = sorter._guides[2]
    original_guid_idx_0 = sorter._guid_idx[0]
    original_guid_idx_2 = sorter._guid_idx[2]
    original_parms_profiles_col_0 = sorter._parms_inters_profiles[:, 0].copy()
    original_parms_profiles_col_2 = sorter._parms_inters_profiles[:, 2].copy()
    original_parms_guides_col_0 = sorter._parms_inters_guides[:, 0].copy()
    original_parms_guides_col_2 = sorter._parms_inters_guides[:, 2].copy()

    sorter.swap_guides(0, 2)

    assert sorter._guides[0] == original_guide_2
    assert sorter._guides[2] == original_guide_0
    assert sorter._guid_idx[0] == original_guid_idx_2
    assert sorter._guid_idx[2] == original_guid_idx_0
    np.testing.assert_array_equal(sorter._parms_inters_profiles[:, 0], original_parms_profiles_col_2)
    np.testing.assert_array_equal(sorter._parms_inters_profiles[:, 2], original_parms_profiles_col_0)
    np.testing.assert_array_equal(sorter._parms_inters_guides[:, 0], original_parms_guides_col_2)
    np.testing.assert_array_equal(sorter._parms_inters_guides[:, 2], original_parms_guides_col_0)


def test_reverse_profile(setup_sorter_data):
    profiles, guides, parms_inters_profiles, parms_inters_guides = setup_sorter_data
    profile_to_reverse_idx = 1
    # Create a specific profile for reversal test
    profiles[profile_to_reverse_idx] = create_linear_bspline_curve(gp_Pnt(0.2, 1, 0), gp_Pnt(0.8, 1, 0))
    parms_inters_profiles[profile_to_reverse_idx, :] = [0.7, 0.5, 0.3]  # Descending order

    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(
        profiles=profiles,
        guides=guides,
        parms_inters_profiles=parms_inters_profiles.copy(),
        parms_inters_guides=parms_inters_guides.copy(),
    )

    original_first_param = sorter._profiles[profile_to_reverse_idx].FirstParameter()
    original_last_param = sorter._profiles[profile_to_reverse_idx].LastParameter()
    original_prof_idx = sorter._prof_idx[profile_to_reverse_idx]

    sorter.reverse_profile(profile_to_reverse_idx)

    assert sorter._profiles[profile_to_reverse_idx].FirstParameter() == original_first_param
    assert sorter._profiles[profile_to_reverse_idx].LastParameter() == original_last_param

    expected_parms = -np.array([0.7, 0.5, 0.3]) + original_first_param + original_last_param
    np.testing.assert_array_almost_equal(sorter._parms_inters_profiles[profile_to_reverse_idx, :], expected_parms)

    assert sorter._prof_idx[profile_to_reverse_idx] == "-" + original_prof_idx


def test_reverse_guide(setup_sorter_data):
    profiles, guides, parms_inters_profiles, parms_inters_guides = setup_sorter_data
    guide_to_reverse_idx = 1
    # Create a specific guide for reversal test
    guides[guide_to_reverse_idx] = create_linear_bspline_curve(gp_Pnt(0.5, 0.3, 0), gp_Pnt(0.5, 0.7, 0))
    parms_inters_guides[:, guide_to_reverse_idx] = [0.6, 0.4, 0.2]  # Descending order

    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(
        profiles=profiles,
        guides=guides,
        parms_inters_profiles=parms_inters_profiles.copy(),
        parms_inters_guides=parms_inters_guides.copy(),
    )

    original_first_param = sorter._guides[guide_to_reverse_idx].FirstParameter()
    original_last_param = sorter._guides[guide_to_reverse_idx].LastParameter()
    original_guid_idx = sorter._guid_idx[guide_to_reverse_idx]

    sorter.reverse_guide(guide_to_reverse_idx)

    assert sorter._guides[guide_to_reverse_idx].FirstParameter() == original_first_param
    assert sorter._guides[guide_to_reverse_idx].LastParameter() == original_last_param

    expected_parms = -np.array([0.6, 0.4, 0.2]) + original_first_param + original_last_param
    np.testing.assert_array_almost_equal(sorter._parms_inters_guides[:, guide_to_reverse_idx], expected_parms)

    assert sorter._guid_idx[guide_to_reverse_idx] == "-" + original_guid_idx


def test_get_start_curve_indices_scenario1(setup_sorter_data):
    profiles, guides, _, _ = setup_sorter_data
    parms_profiles = np.array(
        [[0.1, 0.5, 0.9], [0.6, 0.2, 0.8], [0.7, 0.3, 0.75]], dtype=float
    )
    parms_guides = np.array(
        [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7], [0.9, 0.8, 0.75]], dtype=float
    )
    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(profiles=profiles, guides=guides, parms_inters_profiles=parms_profiles, parms_inters_guides=parms_guides)
    prof_idx, guid_idx, guide_reversed = sorter.get_start_curve_indices()
    assert prof_idx == 0
    assert guid_idx == 0
    assert not guide_reversed


def test_get_start_curve_indices_scenario2_guide_reversed(setup_sorter_data):
    profiles, guides, _, _ = setup_sorter_data
    parms_profiles = np.array(
        [[0.1, 0.5, 0.9], [0.6, 0.2, 0.8], [0.7, 0.3, 0.75]], dtype=float
    )
    parms_guides = np.array(
        [[0.9, 0.2, 0.3], [0.5, 0.6, 0.7], [0.1, 0.8, 0.75]], dtype=float
    )
    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(profiles=profiles, guides=guides, parms_inters_profiles=parms_profiles, parms_inters_guides=parms_guides)
    prof_idx, guid_idx, guide_reversed = sorter.get_start_curve_indices()
    assert prof_idx == 0
    assert guid_idx == 0
    assert guide_reversed


def test_get_start_curve_indices_no_start_found(setup_sorter_data):
    profiles, guides, _, _ = setup_sorter_data
    # These matrices are designed to create a circular dependency
    # where no curve can be identified as the starting point.
    parms_profiles = np.array([
        [0.5, 0.1, 0.9],
        [0.9, 0.5, 0.1],
        [0.1, 0.9, 0.5]
    ])
    parms_guides = np.array([
        [0.1, 0.8, 0.9],
        [0.9, 0.1, 0.8],
        [0.8, 0.9, 0.1]
    ])
    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(profiles=profiles, guides=guides, parms_inters_profiles=parms_profiles, parms_inters_guides=parms_guides)
    with pytest.raises(error, match="Cannot find starting curves of curve network."):
        sorter.get_start_curve_indices()


def test_perform_already_sorted(setup_sorter_data):
    profiles, guides, _, _ = setup_sorter_data
    parms_profiles = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=float
    )
    parms_guides = np.array(
        [[0.1, 0.4, 0.7], [0.2, 0.5, 0.8], [0.3, 0.6, 0.9]], dtype=float
    )

    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(
        profiles=profiles,
        guides=guides,
        parms_inters_profiles=parms_profiles.copy(),
        parms_inters_guides=parms_guides.copy(),
    )
    sorter.perform()

    assert sorter._has_performed
    np.testing.assert_array_equal(sorter._parms_inters_profiles, parms_profiles)
    np.testing.assert_array_equal(sorter._parms_inters_guides, parms_guides)
    assert sorter.ProfileIndices() == ["0", "1", "2"]
    assert sorter.GuideIndices() == ["0", "1", "2"]
    assert sorter._profiles[0].FirstParameter() == 0.0  # Check original state
    assert sorter._guides[0].FirstParameter() == 0.0  # Check original state


def test_perform_needs_sorting_and_reversal_complex(setup_sorter_data):
    profiles, guides, _, _ = setup_sorter_data

    # This test case is designed to trigger a complex sorting and reversal scenario.
    # The matrices are constructed so that the algorithm must:
    # 1. Identify P2 (profile 2) and G1 (guide 1) as the starting corner.
    # 2. Recognize that G1 needs to be reversed.
    # 3. Swap P2 to the first position.
    # 4. Swap the reversed G1 to the first position.
    # 5. Sort the remaining profiles and guides based on proximity.

    # P0 params: [0.5, 0.6, 0.7] -> min at index 0
    # P1 params: [0.4, 0.3, 0.2] -> min at index 2
    # P2 params: [0.8, 0.1, 0.9] -> min at index 1
    parms_profiles = np.array([
        [0.5, 0.6, 0.7],
        [0.4, 0.3, 0.2],
        [0.8, 0.1, 0.9]
    ])

    # G0 params: [0.5, 0.4, 0.6] -> min at row 1, max at row 2
    # G1 params: [0.2, 0.3, 1.0] -> min at row 0, max at row 2
    # G2 params: [0.8, 0.7, 0.9] -> min at row 1, max at row 2
    # This setup ensures that for k=2, l=1 is chosen, and max_col_index(guides, 1) is 2.
    parms_guides = np.array([
        [0.5, 0.2, 0.8],
        [0.4, 0.3, 0.7],
        [0.6, 1.0, 0.9]
    ])

    sorter = CurveNetworkSorter(
        profiles=profiles,
        guides=guides,
        parms_inters_profiles=parms_profiles,
        parms_inters_guides=parms_guides,
    )
    sorter.perform()

    assert sorter._has_performed

    # Expected order after starting with P1 and sorting by proximity: P1, P0, P2
    assert sorter.ProfileIndices() == ["1", "-0", "-2"]

    # Expected order: G2 is moved to the start. G0 and G1 are sorted.
    assert sorter.GuideIndices() == ["2", "1", "0"]



if __name__ == "__main__":
    # pytest.main([f'{__file__}::test_get_start_curve_indices_no_start_found', "-v"])
    pytest.main([f'{__file__}', "-v"])
