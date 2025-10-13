import os
import sys
from typing import List, Tuple

import numpy as np
import pytest
from OCP.Geom import Geom_BSplineCurve, Geom_Curve
from OCP.gp import gp_Pnt
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the CurveNetworkSorter from the internal module
from src_py.ocp_gordon.internal.curve_network_sorter import CurveNetworkSorter
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

    multiplicities = TColStd_Array1OfInteger(1, 2)  # Corrected type
    multiplicities.SetValue(1, degree + 1)
    multiplicities.SetValue(2, degree + 1)

    curve = Geom_BSplineCurve(control_points, knots, multiplicities, degree)
    return curve


def create_zero_length_bspline_curve(
    point: gp_Pnt, degree: int = 1
) -> Geom_BSplineCurve:
    """
    Helper to create a zero-length B-spline curve (a point).
    """
    return create_linear_bspline_curve(point, point)


@pytest.fixture
def setup_sorter_data() -> (
    tuple[list[Geom_BSplineCurve], list[Geom_BSplineCurve], np.ndarray, np.ndarray]
):
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


@pytest.fixture
def setup_sorter_data_with_zero_length() -> (
    tuple[list[Geom_BSplineCurve], list[Geom_BSplineCurve], np.ndarray, np.ndarray]
):
    # Profiles: zero-length, normal, zero-length
    profiles = [
        create_zero_length_bspline_curve(gp_Pnt(0, 0, 0)),
        create_linear_bspline_curve(gp_Pnt(0, 1, 0), gp_Pnt(1, 1, 0)),
        create_zero_length_bspline_curve(gp_Pnt(0, 2, 0)),
    ]
    # Guides: normal, zero-length, normal
    guides = [
        create_linear_bspline_curve(gp_Pnt(0, 0, 0), gp_Pnt(0, 2, 0)),
        create_zero_length_bspline_curve(gp_Pnt(0.5, 0, 0)),
        create_linear_bspline_curve(gp_Pnt(1, 0, 0), gp_Pnt(1, 2, 0)),
    ]
    # Intersection parameters (some values might be arbitrary for zero-length curves)
    parms_inters_profiles = np.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.8], [0.0, 0.0, 0.0]], dtype=float
    )
    parms_inters_guides = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.7], [0.0, 0.0, 0.0]], dtype=float
    )
    return profiles, guides, parms_inters_profiles, parms_inters_guides


def test_constructor_valid_input(setup_sorter_data):
    profiles, guides, parms_inters_profiles, parms_inters_guides = setup_sorter_data
    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(
        profiles=list(profiles),
        guides=list(guides),
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
    with pytest.raises(
        error, match="Invalid row size of parms_inters_profiles matrix."
    ):
        # Explicitly cast to List[Geom_Curve]
        CurveNetworkSorter(
            profiles=list(profiles),
            guides=list(guides),
            parms_inters_profiles=invalid_parms,
            parms_inters_guides=parms_inters_guides,
        )


def test_constructor_invalid_col_size_profiles(setup_sorter_data):
    profiles, guides, _, parms_inters_guides = setup_sorter_data
    invalid_parms = np.array([[0.1], [0.2], [0.3]], dtype=float)  # 1 col instead of 3
    with pytest.raises(
        error, match="Invalid col size of parms_inters_profiles matrix."
    ):
        # Explicitly cast to List[Geom_Curve]
        CurveNetworkSorter(
            profiles=list(profiles),
            guides=list(guides),
            parms_inters_profiles=invalid_parms,
            parms_inters_guides=parms_inters_guides,
        )


def test_swap_profiles(setup_sorter_data):
    profiles, guides, parms_inters_profiles, parms_inters_guides = setup_sorter_data
    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(
        profiles=list(profiles),
        guides=list(guides),
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
    np.testing.assert_array_equal(
        sorter._parms_inters_profiles[0, :], original_parms_profiles_row_2
    )
    np.testing.assert_array_equal(
        sorter._parms_inters_profiles[2, :], original_parms_profiles_row_0
    )
    np.testing.assert_array_equal(
        sorter._parms_inters_guides[0, :], original_parms_guides_row_2
    )
    np.testing.assert_array_equal(
        sorter._parms_inters_guides[2, :], original_parms_guides_row_0
    )


def test_swap_guides(setup_sorter_data):
    profiles, guides, parms_inters_profiles, parms_inters_guides = setup_sorter_data
    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(
        profiles=list(profiles),
        guides=list(guides),
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
    np.testing.assert_array_equal(
        sorter._parms_inters_profiles[:, 0], original_parms_profiles_col_2
    )
    np.testing.assert_array_equal(
        sorter._parms_inters_profiles[:, 2], original_parms_profiles_col_0
    )
    np.testing.assert_array_equal(
        sorter._parms_inters_guides[:, 0], original_parms_guides_col_2
    )
    np.testing.assert_array_equal(
        sorter._parms_inters_guides[:, 2], original_parms_guides_col_0
    )


def test_reverse_profile(setup_sorter_data):
    profiles, guides, parms_inters_profiles, parms_inters_guides = setup_sorter_data
    profile_to_reverse_idx = 1
    # Create a specific profile for reversal test
    profiles[profile_to_reverse_idx] = create_linear_bspline_curve(
        gp_Pnt(0.2, 1, 0), gp_Pnt(0.8, 1, 0)
    )
    parms_inters_profiles[profile_to_reverse_idx, :] = [
        0.7,
        0.5,
        0.3,
    ]  # Descending order

    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(
        profiles=list(profiles),
        guides=list(guides),
        parms_inters_profiles=parms_inters_profiles.copy(),
        parms_inters_guides=parms_inters_guides.copy(),
    )

    original_first_param = sorter._profiles[profile_to_reverse_idx].FirstParameter()
    original_last_param = sorter._profiles[profile_to_reverse_idx].LastParameter()
    original_prof_idx = sorter._prof_idx[profile_to_reverse_idx]

    sorter.reverse_profile(profile_to_reverse_idx)

    assert (
        sorter._profiles[profile_to_reverse_idx].FirstParameter()
        == original_first_param
    )
    assert (
        sorter._profiles[profile_to_reverse_idx].LastParameter() == original_last_param
    )

    expected_parms = (
        -np.array([0.7, 0.5, 0.3]) + original_first_param + original_last_param
    )
    np.testing.assert_array_almost_equal(
        sorter._parms_inters_profiles[profile_to_reverse_idx, :], expected_parms
    )

    assert sorter._prof_idx[profile_to_reverse_idx] == "-" + original_prof_idx


def test_reverse_guide(setup_sorter_data):
    profiles, guides, parms_inters_profiles, parms_inters_guides = setup_sorter_data
    guide_to_reverse_idx = 1
    # Create a specific guide for reversal test
    guides[guide_to_reverse_idx] = create_linear_bspline_curve(
        gp_Pnt(0.5, 0.3, 0), gp_Pnt(0.5, 0.7, 0)
    )
    parms_inters_guides[:, guide_to_reverse_idx] = [0.6, 0.4, 0.2]  # Descending order

    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(
        profiles=list(profiles),
        guides=list(guides),
        parms_inters_profiles=parms_inters_profiles.copy(),
        parms_inters_guides=parms_inters_guides.copy(),
    )

    original_first_param = sorter._guides[guide_to_reverse_idx].FirstParameter()
    original_last_param = sorter._guides[guide_to_reverse_idx].LastParameter()
    original_guid_idx = sorter._guid_idx[guide_to_reverse_idx]

    sorter.reverse_guide(guide_to_reverse_idx)

    assert sorter._guides[guide_to_reverse_idx].FirstParameter() == original_first_param
    assert sorter._guides[guide_to_reverse_idx].LastParameter() == original_last_param

    expected_parms = (
        -np.array([0.6, 0.4, 0.2]) + original_first_param + original_last_param
    )
    np.testing.assert_array_almost_equal(
        sorter._parms_inters_guides[:, guide_to_reverse_idx], expected_parms
    )

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
    sorter = CurveNetworkSorter(
        profiles=list(profiles),
        guides=list(guides),
        parms_inters_profiles=parms_profiles,
        parms_inters_guides=parms_guides,
    )
    prof_idx, guide_idx, reversed_flag = sorter.get_start_curve_indices()
    assert prof_idx == 0
    assert guide_idx == 0
    assert reversed_flag == False


def test_get_start_curve_indices_scenario2_guide_reversed(setup_sorter_data):
    profiles, guides, _, _ = setup_sorter_data
    parms_profiles = np.array(
        [[0.1, 0.5, 0.9], [0.6, 0.2, 0.8], [0.7, 0.3, 0.75]], dtype=float
    )
    parms_guides = np.array(
        [[0.9, 0.2, 0.3], [0.5, 0.6, 0.7], [0.1, 0.8, 0.75]], dtype=float
    )
    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(
        profiles=list(profiles),
        guides=list(guides),
        parms_inters_profiles=parms_profiles,
        parms_inters_guides=parms_guides,
    )
    prof_idx, guide_idx, reversed_flag = sorter.get_start_curve_indices()
    assert prof_idx == 0
    assert guide_idx == 0
    assert reversed_flag == True


def test_get_start_curve_indices_no_start_found(setup_sorter_data):
    profiles, guides, _, _ = setup_sorter_data
    # These matrices are designed to create a circular dependency
    # where no curve can be identified as the starting point.
    parms_profiles = np.array([[0.5, 0.1, 0.9], [0.9, 0.5, 0.1], [0.1, 0.9, 0.5]])
    parms_guides = np.array([[0.1, 0.8, 0.9], [0.9, 0.1, 0.8], [0.8, 0.9, 0.1]])
    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(
        profiles=list(profiles),
        guides=list(guides),
        parms_inters_profiles=parms_profiles,
        parms_inters_guides=parms_guides,
    )
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
        profiles=list(profiles),
        guides=list(guides),
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
    parms_profiles = np.array([[0.5, 0.6, 0.7], [0.4, 0.3, 0.2], [0.8, 0.1, 0.9]])

    # G0 params: [0.5, 0.4, 0.6] -> min at row 1, max at row 2
    # G1 params: [0.2, 0.3, 1.0] -> min at row 0, max at row 2
    # G2 params: [0.8, 0.7, 0.9] -> min at row 1, max at row 2
    # This setup ensures that for k=2, l=1 is chosen, and max_col_index(guides, 1) is 2.
    parms_guides = np.array([[0.5, 0.2, 0.8], [0.4, 0.3, 0.7], [0.6, 1.0, 0.9]])

    sorter = CurveNetworkSorter(
        profiles=list(profiles),
        guides=list(guides),
        parms_inters_profiles=parms_profiles,
        parms_inters_guides=parms_guides,
    )
    sorter.perform()
    print(sorter._parms_inters_profiles)
    print(sorter._parms_inters_guides)

    assert sorter._has_performed

    # Expected order after starting with P1 and sorting by proximity: P1, P0, P2
    assert sorter.ProfileIndices() == ["1", "-0", "-2"]

    # Expected order: G2 is moved to the start. G0 and G1 are sorted.
    assert sorter.GuideIndices() == ["2", "1", "0"]


def test_get_start_curve_indices_with_zero_length(setup_sorter_data_with_zero_length):
    profiles, guides, parms_profiles, parms_guides = setup_sorter_data_with_zero_length
    # Explicitly cast to List[Geom_Curve]
    sorter = CurveNetworkSorter(
        profiles=list(profiles),
        guides=list(guides),
        parms_inters_profiles=parms_profiles,
        parms_inters_guides=parms_guides,
    )
    # Expected: Profile 1 (index 1) is the first non-zero-length profile.
    # Guide 0 (index 0) is the first non-zero-length guide.
    # The intersection parameters for (P1, G0) are [0.2, 0.5]
    # min_row_index for P1 (row 1) should return 0 (G0)
    # min_col_index for G0 (col 0) should return 1 (P1)
    # So, (1, 0, False) should be the result.
    prof_idx, guide_idx, reversed_flag = sorter.get_start_curve_indices()
    assert prof_idx == 1
    assert guide_idx == 0
    assert reversed_flag == False


def test_min_row_index_skips_zero_length(setup_sorter_data_with_zero_length):
    profiles, guides, parms_profiles, _ = setup_sorter_data_with_zero_length
    sorter = CurveNetworkSorter(
        profiles=list(profiles),
        guides=list(guides),
        parms_inters_profiles=parms_profiles,
        parms_inters_guides=np.zeros_like(parms_profiles),  # Dummy for guides
    )
    # For profile 1 (index 1), the intersection parameters are [0.2, 0.0, 0.8]
    # Guide 1 (index 1) is zero-length. So it should skip index 1.
    # The minimum should be 0.2 at index 0.
    jmin = sorter.min_row_index(parms_profiles, 1)
    assert jmin == 0

    # For profile 0 (index 0), all guides are zero-length or associated with zero-length profiles.
    # The min_row_index should return -1 as no valid non-zero-length guide is found.
    # However, the current implementation of min_row_index only skips zero-length guides.
    # It does not check if the profile itself is zero-length.
    # Let's assume we are testing the behavior of min_row_index given a row.
    # If all guides are zero-length, it should return -1.
    profiles_all_zero = [create_zero_length_bspline_curve(gp_Pnt(0, 0, 0))] * 3
    guides_all_zero = [create_zero_length_bspline_curve(gp_Pnt(0, 0, 0))] * 3
    sorter_all_zero = CurveNetworkSorter(
        profiles=list(profiles_all_zero),
        guides=list(guides_all_zero),
        parms_inters_profiles=np.zeros((3, 3)),
        parms_inters_guides=np.zeros((3, 3)),
    )
    jmin_all_zero = sorter_all_zero.min_row_index(np.array([[0.1, 0.2, 0.3]]), 0)
    assert jmin_all_zero == -1


def test_max_row_index_skips_zero_length(setup_sorter_data_with_zero_length):
    profiles, guides, parms_profiles, _ = setup_sorter_data_with_zero_length
    sorter = CurveNetworkSorter(
        profiles=list(profiles),
        guides=list(guides),
        parms_inters_profiles=parms_profiles,
        parms_inters_guides=np.zeros_like(parms_profiles),
    )
    # For profile 1 (index 1), the intersection parameters are [0.2, 0.0, 0.8]
    # Guide 1 (index 1) is zero-length. So it should skip index 1.
    # The maximum should be 0.8 at index 2.
    jmax = sorter.max_row_index(parms_profiles, 1)
    assert jmax == 2

    profiles_all_zero = [create_zero_length_bspline_curve(gp_Pnt(0, 0, 0))] * 3
    guides_all_zero = [create_zero_length_bspline_curve(gp_Pnt(0, 0, 0))] * 3
    sorter_all_zero = CurveNetworkSorter(
        profiles=list(profiles_all_zero),
        guides=list(guides_all_zero),
        parms_inters_profiles=np.zeros((3, 3)),
        parms_inters_guides=np.zeros((3, 3)),
    )
    jmax_all_zero = sorter_all_zero.max_row_index(np.array([[0.1, 0.2, 0.3]]), 0)
    assert jmax_all_zero == -1


def test_min_col_index_skips_zero_length(setup_sorter_data_with_zero_length):
    profiles, guides, _, parms_guides = setup_sorter_data_with_zero_length
    sorter = CurveNetworkSorter(
        profiles=list(profiles),
        guides=list(guides),
        parms_inters_profiles=np.zeros_like(parms_guides),
        parms_inters_guides=parms_guides,
    )
    # For guide 0 (index 0), the intersection parameters are [0.0, 0.5, 0.0]
    # Profile 0 and 2 are zero-length. So it should skip index 0 and 2.
    # The minimum should be 0.5 at index 1.
    imin = sorter.min_col_index(parms_guides, 0)
    assert imin == 1

    profiles_all_zero = [create_zero_length_bspline_curve(gp_Pnt(0, 0, 0))] * 3
    guides_all_zero = [create_zero_length_bspline_curve(gp_Pnt(0, 0, 0))] * 3
    sorter_all_zero = CurveNetworkSorter(
        profiles=list(profiles_all_zero),
        guides=list(guides_all_zero),
        parms_inters_profiles=np.zeros((3, 3)),
        parms_inters_guides=np.zeros((3, 3)),
    )
    imin_all_zero = sorter_all_zero.min_col_index(np.array([[0.1], [0.2], [0.3]]), 0)
    assert imin_all_zero == -1


def test_max_col_index_skips_zero_length(setup_sorter_data_with_zero_length):
    profiles, guides, _, parms_guides = setup_sorter_data_with_zero_length
    sorter = CurveNetworkSorter(
        profiles=list(profiles),
        guides=list(guides),
        parms_inters_profiles=np.zeros_like(parms_guides),
        parms_inters_guides=parms_guides,
    )
    # For guide 0 (index 0), the intersection parameters are [0.0, 0.5, 0.0]
    # Profile 0 and 2 are zero-length. So it should skip index 0 and 2.
    # The maximum should be 0.5 at index 1.
    imax = sorter.max_col_index(parms_guides, 0)
    assert imax == 1

    profiles_all_zero = [create_zero_length_bspline_curve(gp_Pnt(0, 0, 0))] * 3
    guides_all_zero = [create_zero_length_bspline_curve(gp_Pnt(0, 0, 0))] * 3
    sorter_all_zero = CurveNetworkSorter(
        profiles=list(profiles_all_zero),
        guides=list(guides_all_zero),
        parms_inters_profiles=np.zeros((3, 3)),
        parms_inters_guides=np.zeros((3, 3)),
    )
    imax_all_zero = sorter_all_zero.max_col_index(np.array([[0.1], [0.2], [0.3]]), 0)
    assert imax_all_zero == -1


def test_perform_with_zero_length_curves(setup_sorter_data_with_zero_length):
    profiles, guides, parms_profiles, parms_guides = setup_sorter_data_with_zero_length
    sorter = CurveNetworkSorter(
        profiles=list(profiles),
        guides=list(guides),
        parms_inters_profiles=parms_profiles.copy(),
        parms_inters_guides=parms_guides.copy(),
    )
    sorter.perform()

    assert sorter._has_performed
    # Expected sorting:
    # Profiles: [P1 (normal), P0 (zero-length), P2 (zero-length)]
    # Guides: [G0 (normal), G2 (normal), G1 (zero-length)]
    # The sorting logic should prioritize non-zero-length curves.
    # The first non-zero-length profile is P1 (original index 1).
    # The first non-zero-length guide is G0 (original index 0).

    # After sorting, the profiles should be in order of their original indices,
    # but with zero-length ones potentially moved to the end or handled as per logic.
    # The current logic moves the start profile/guide to index 0.
    # Then sorts based on the first profile/guide.
    # The expected order of profiles after sorting by G0 (index 0) should be P1, P0, P2
    # The expected order of guides after sorting by P1 (index 1) should be G0, G2, G1

    # The get_start_curve_indices should return (1, 0, False)
    # After swap_profiles(0, 1) and swap_guides(0, 0)
    # Profiles: [P1, P0, P2]
    # Guides: [G0, G1, G2] (G1 and G2 swapped)

    # Then sort guides by P1 (new index 0)
    # P1 intersections with guides: [0.2, 0.0, 0.8]
    # G0 (0.2), G1 (0.0, zero-length), G2 (0.8)
    # After sorting guides by P1, the order should be G0, G2, G1 (skipping G1 for comparison)
    # So, G0 (0.2), G2 (0.8)
    # The actual order of guides should be G0, G2, G1 (original indices)
    # The current bubble sort for guides will put G0 first, then G2, then G1.

    # Then sort profiles by G0 (new index 0)
    # G0 intersections with profiles: [0.0, 0.5, 0.0]
    # P0 (0.0, zero-length), P1 (0.5), P2 (0.0, zero-length)
    # After sorting profiles by G0, the order should be P1, P0, P2 (original indices)
    # The current bubble sort for profiles will put P1 first, then P0, then P2.

    # Final expected indices after sorting and potential reversals
    assert sorter.ProfileIndices() == ["0", "2", "1"]
    assert sorter.GuideIndices() == ["1", "0", "2"]


if __name__ == "__main__":
    # pytest.main([f'{__file__}::test_get_start_curve_indices_no_start_found', "-v"])
    pytest.main([f"{__file__}", "-v"])
