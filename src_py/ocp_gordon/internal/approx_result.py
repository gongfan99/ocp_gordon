"""
SPDX-License-Identifier: Apache-20
SPDX-FileCopyrightText: 2020 German Aerospace Center (DLR)

Created: 2020-07-15 Martin Siggel <Martin.Siggel@dlr.de>
"""

from OCP.Geom import Geom_BSplineCurve
from typing import Optional

class ApproxResult:
    """
    Structure to hold the result of a B-spline approximation/interpolation.
    """
    def __init__(self, curve: Geom_BSplineCurve | None = None, error: float = 0.0):
        self.curve = curve
        self.error = error
