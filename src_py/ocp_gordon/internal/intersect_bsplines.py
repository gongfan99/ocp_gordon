import math
import sys
from typing import Any, Callable, TypedDict

import numpy as np
from OCP.Geom import Geom_BSplineCurve, Geom_Curve
from OCP.gp import gp_Pnt, gp_Vec
from scipy.optimize import OptimizeResult, minimize_scalar

from .misc import (
    Standard_Real,
    clone_bspline,
    math_BFGS,
    math_MultipleVarFunctionWithGradient,
    math_Vector,
)

# --- Helper functions and classes from C++ ---


# def maxval(v1, v2):
#     return v1 if v1 > v2 else v2


# def sqr(v):
#     return v * v


def minCoords(p1: gp_Pnt, p2: gp_Pnt) -> gp_Pnt:
    return gp_Pnt(min(p1.X(), p2.X()), min(p1.Y(), p2.Y()), min(p1.Z(), p2.Z()))


def maxCoords(p1: gp_Pnt, p2: gp_Pnt) -> gp_Pnt:
    return gp_Pnt(max(p1.X(), p2.X()), max(p1.Y(), p2.Y()), max(p1.Z(), p2.Z()))


class Intervall:
    def __init__(self, mmin: float, mmax: float):
        self.min = mmin
        self.max = mmax

    def __eq__(self, other):
        EPS = 1e-15
        return abs(self.min - other.min) < EPS and abs(self.max - other.max) < EPS


class BoundingBox:
    def __init__(self, curve_or_other: "Geom_BSplineCurve | BoundingBox"):
        if isinstance(curve_or_other, Geom_BSplineCurve):
            curve = curve_or_other
            self.range = Intervall(curve.FirstParameter(), curve.LastParameter())
            np_array = np.zeros((curve.NbPoles(), 3), dtype=np.float64)
            for i in range(1, curve.NbPoles() + 1):
                p = curve.Pole(i)
                np_array[i - 1] = [
                    p.X(),
                    p.Y(),
                    p.Z(),
                ]
            temp = np.min(np_array, axis=0)
            self.low = gp_Pnt(temp[0], temp[1], temp[2])
            temp = np.max(np_array, axis=0)
            self.high = gp_Pnt(temp[0], temp[1], temp[2])
        else:
            other = curve_or_other
            self.range = Intervall(other.range.min, other.range.max)
            self.low = gp_Pnt(other.low.XYZ())
            self.high = gp_Pnt(other.high.XYZ())

    def Intersects(self, other: "BoundingBox", eps: float) -> bool:
        min_coords = maxCoords(self.low, other.low)
        max_coords = minCoords(self.high, other.high)
        return (
            (min_coords.X() < max_coords.X() + eps)
            and (min_coords.Y() < max_coords.Y() + eps)
            and (min_coords.Z() < max_coords.Z() + eps)
        )

    def Merge(self, other: "BoundingBox") -> "BoundingBox":
        self.range.max = other.range.max
        self.high = maxCoords(self.high, other.high)
        self.low = minCoords(self.low, other.low)
        return self

    def __eq__(self, other):
        return self.range == other.range


def curvature(curve: Geom_BSplineCurve) -> float:
    len_curve = curve.Pole(1).Distance(curve.Pole(curve.NbPoles()))
    total = 0.0
    for i in range(1, curve.NbPoles()):
        p1 = curve.Pole(i)
        p2 = curve.Pole(i + 1)
        dist = p1.Distance(p2)
        total += dist

    if abs(len_curve) < 1e-15:
        return 1.0 if total < 1e-15 else 1e9
    return max(total / len_curve, 1.0)


class BoundingBoxPair:
    def __init__(self, b1: BoundingBox, b2: BoundingBox):
        self.b1 = b1
        self.b2 = b2


class BSplineAlgorithms:
    @staticmethod
    def trimCurve(
        curve: Geom_BSplineCurve, first: float, last: float
    ) -> Geom_BSplineCurve:
        new_curve = clone_bspline(curve)
        new_curve.Segment(first, last)
        return new_curve


# --- Main function and classes to translate ---


# math_MultipleVarFunctionWithGradient
class CurveCurveDistanceObjective(math_MultipleVarFunctionWithGradient):
    def __init__(self, c1: Geom_Curve, c2: Geom_Curve):
        # Call the parent constructor without arguments, as math_MultipleVarFunction has no __init__
        super().__init__()
        self.m_c1 = c1
        self.m_c2 = c2

    def NbVariables(self) -> int:
        return 2

    def Value(self, X: math_Vector, F: Standard_Real) -> bool:
        G = math_Vector(1, 2)
        self.Values(X, F, G)
        return True  # The original C++ returns true, and F is an output parameter

    def Gradient(
        self, X: math_Vector, G: math_Vector
    ) -> bool:  # Gradient also takes G as an output parameter
        F_val = Standard_Real(0.0)  # Create a temporary Standard_Real for F
        self.Values(X, F_val, G)
        return True

    # C++ uses 0.5 * (math.sin(z) + 1) as the activate function to limit the u, v value to [0, 1]
    # But it is found that the optimization does not converge in some rare cases.
    # Here we use f(z)=z as the activate function which seems more stable for optimization.
    @staticmethod
    def activate(z: float) -> float:
        return z

    @staticmethod
    def d_activate(z: float) -> float:
        return 1

    # @staticmethod
    # def activate(z: float) -> float:
    #     return 0.5 * math.sin(z) + 0.5

    # @staticmethod
    # def d_activate(z: float) -> float:
    #     return 0.5 * math.cos(z)

    # @staticmethod
    # def activate(z: float) -> float:
    #     return 1.25 * math.atan(z) / math.pi + 0.5

    # @staticmethod
    # def d_activate(z: float) -> float:
    #     return 1.25 / (math.pi * (1.0 + z**2))

    def getUParam(self, x0: float) -> float:
        umin = self.m_c1.FirstParameter()
        umax = self.m_c1.LastParameter()
        return self.activate(x0) * (umax - umin) + umin

    def getVParam(self, x1: float) -> float:
        vmin = self.m_c2.FirstParameter()
        vmax = self.m_c2.LastParameter()
        return self.activate(x1) * (vmax - vmin) + vmin

    def d_getUParam(self, x0: float) -> float:
        umin = self.m_c1.FirstParameter()
        umax = self.m_c1.LastParameter()
        return self.d_activate(x0) * (umax - umin)

    def d_getVParam(self, x1: float) -> float:
        vmin = self.m_c2.FirstParameter()
        vmax = self.m_c2.LastParameter()
        return self.d_activate(x1) * (vmax - vmin)

    def Values(self, X: math_Vector, F: Standard_Real, G: math_Vector) -> bool:
        u = self.getUParam(X.Value(1))
        v = self.getVParam(X.Value(2))

        p1 = gp_Pnt(0, 0, 0)
        p2 = gp_Pnt(0, 0, 0)
        d1_vec = gp_Vec(0, 0, 0)
        d2_vec = gp_Vec(0, 0, 0)

        self.m_c1.D1(u, p1, d1_vec)
        self.m_c2.D1(v, p2, d2_vec)

        diff = gp_Vec(p2, p1)  # p1 - p2
        F.value = diff.SquareMagnitude()

        G.SetValue(
            1,
            2.0 * diff.Dot(d1_vec) * self.d_getUParam(X.Value(1)),
        )
        G.SetValue(
            2,
            -2.0 * diff.Dot(d2_vec) * self.d_getVParam(X.Value(2)),
        )

        return True


def find_split_point(curve: Geom_BSplineCurve):
    """
    Find the max inner multiplicity. If it is higher than 1, return the knot.
    otherwise return the mid point
    """
    mid_point = 0.5 * (curve.FirstParameter() + curve.LastParameter())
    if curve.IsCN(curve.Degree() - 1):
        return mid_point

    max_inner_mult_index = 2
    for i in range(3, curve.NbKnots()):
        if curve.Multiplicity(i) > curve.Multiplicity(max_inner_mult_index):
            max_inner_mult_index = i

    if curve.Multiplicity(max_inner_mult_index) > 1:
        return curve.Knot(max_inner_mult_index)

    return mid_point


def getRangesOfIntersection(
    curve1: Geom_BSplineCurve, curve2: Geom_BSplineCurve, tolerance: float
) -> list[BoundingBoxPair]:
    h1 = BoundingBox(curve1)
    h2 = BoundingBox(curve2)

    if not h1.Intersects(h2, tolerance):
        return []

    c1_curvature = curvature(curve1)
    c2_curvature = curvature(curve2)
    max_curvature = 1.0005

    if c1_curvature <= max_curvature and c2_curvature <= max_curvature:
        return [BoundingBoxPair(h1, h2)]

    curve1MidParm = find_split_point(curve1)
    curve2MidParm = find_split_point(curve2)

    results = []

    if c1_curvature > max_curvature and c2_curvature > max_curvature:
        c11 = BSplineAlgorithms.trimCurve(
            curve1, curve1.FirstParameter(), curve1MidParm
        )
        c12 = BSplineAlgorithms.trimCurve(curve1, curve1MidParm, curve1.LastParameter())

        c21 = BSplineAlgorithms.trimCurve(
            curve2, curve2.FirstParameter(), curve2MidParm
        )
        c22 = BSplineAlgorithms.trimCurve(curve2, curve2MidParm, curve2.LastParameter())

        results.extend(getRangesOfIntersection(c11, c21, tolerance))
        results.extend(getRangesOfIntersection(c11, c22, tolerance))
        results.extend(getRangesOfIntersection(c12, c21, tolerance))
        results.extend(getRangesOfIntersection(c12, c22, tolerance))

    elif c1_curvature <= max_curvature and max_curvature < c2_curvature:
        c21 = BSplineAlgorithms.trimCurve(
            curve2, curve2.FirstParameter(), curve2MidParm
        )
        c22 = BSplineAlgorithms.trimCurve(curve2, curve2MidParm, curve2.LastParameter())

        results.extend(getRangesOfIntersection(curve1, c21, tolerance))
        results.extend(getRangesOfIntersection(curve1, c22, tolerance))

    elif c2_curvature <= max_curvature and max_curvature < c1_curvature:
        c11 = BSplineAlgorithms.trimCurve(
            curve1, curve1.FirstParameter(), curve1MidParm
        )
        c12 = BSplineAlgorithms.trimCurve(curve1, curve1MidParm, curve1.LastParameter())

        results.extend(getRangesOfIntersection(c11, curve2, tolerance))
        results.extend(getRangesOfIntersection(c12, curve2, tolerance))

    return results


def replace_adjacent_in_list(
    avoid_u_values: list[float],
    list_obj: list[BoundingBox],
    is_adjacent_func: Callable[[list[float], BoundingBox, BoundingBox], bool],
    merge_func: Callable[[BoundingBox, BoundingBox], BoundingBox],
):
    i = 0
    while i < len(list_obj):
        next_i = i + 1
        if next_i >= len(list_obj):
            break

        if is_adjacent_func(avoid_u_values, list_obj[i], list_obj[next_i]):
            merged_val = merge_func(list_obj[i], list_obj[next_i])
            list_obj.pop(i)
            list_obj.pop(i)
            list_obj.insert(i, merged_val)
        else:
            i = next_i


def merge_boxes(b1: BoundingBox, b2: BoundingBox) -> BoundingBox:
    new_box = BoundingBox(b1)
    #             self.range = Intervall(other.range.min, other.range.max)
    #             self.low = gp_Pnt(other.low.XYZ())
    #             self.high = gp_Pnt(other.high.XYZ())
    new_box.Merge(b2)
    return new_box


# add two functions is_point_on_line_segment and line_line_intersection_3d
# to triage the intersection to avoid expensive minimizer
# and find good intial solution
# They are not in C++ code
def is_point_on_line_segment(
    point: gp_Pnt,
    segment_start: gp_Pnt,
    segment_end: gp_Pnt,
    max_curvature: float = 1.0005,
) -> tuple[float, bool]:
    """
    Checks if a 3D point lies on a 3D line segment which
    is the chord of a B-spline with the max_curvature.
    t is the parameter for the line segment in the range of [0, 1].
    """
    vec_segment = gp_Vec(segment_start, segment_end)
    vec_point_from_start = gp_Vec(segment_start, point)

    # t = (AP . AB) / |AB|^2
    line_length = vec_segment.Magnitude()
    if line_length < 1e-9:
        return 0, vec_point_from_start.SquareMagnitude() < 1e-9
    t = vec_point_from_start.Dot(vec_segment) / (line_length * line_length)

    possible_intersect = True

    # use a circular arc to estimate the max possible distance
    rel_sagitta = 0.6123724 * math.sqrt(max_curvature - 1)
    max_distance = 1.2 * rel_sagitta * line_length  # 1.2 for margin

    # check the range of t
    if t * line_length < -max_distance or t * line_length > line_length + max_distance:
        possible_intersect = False

    # Check the distance between the closest points
    closest_point_on_line = gp_Pnt(segment_start.XYZ())
    closest_point_on_line.BaryCenter(1 - t, segment_end, t)

    distance_to_line = point.Distance(closest_point_on_line)
    if distance_to_line > max_distance:
        possible_intersect = False

    return t, possible_intersect


def line_line_intersection_3d(
    p1_start: gp_Pnt,
    p1_end: gp_Pnt,
    p2_start: gp_Pnt,
    p2_end: gp_Pnt,
    max_curvature: float = 1.0005,
) -> tuple[float, float, bool]:
    """
    Finds the parameters (t, s) for the closest points on two 3D line segments which
    are the chords of possibly intersected two B-splines with the max_curvature.
    t is the parameter for line 1, s for line 2. Both are in the range of [0, 1].
    """
    u = gp_Vec(p1_start, p1_end)
    v = gp_Vec(p2_start, p2_end)
    w = gp_Vec(p1_start, p2_start)

    a = u.Dot(u)  # |u|^2
    b = u.Dot(v)
    c = v.Dot(v)  # |v|^2
    d = u.Dot(w)
    e = v.Dot(w)

    denom = a * c - b * b

    if abs(denom) < 1e-12:  # Lines are parallel or collinear
        t = 0.5
        s = 0.5
    else:
        t = (c * d - b * e) / denom
        s = (b * d - a * e) / denom

    possible_intersect = True

    # use a circular arc to estimate the max possible distance between two lines
    p1_mag = math.sqrt(a)
    p2_mag = math.sqrt(c)
    rel_sagitta = 0.6123724 * math.sqrt(max_curvature - 1)
    max_distance = 1.2 * rel_sagitta * (p1_mag + p2_mag)  # 1.2 for margin

    # check the range of t, s
    if t * p1_mag < -max_distance or t * p1_mag > p1_mag + max_distance:
        possible_intersect = False

    if s * p2_mag < -max_distance or s * p2_mag > p2_mag + max_distance:
        possible_intersect = False

    # Check the distance between the closest points
    closest_p1 = gp_Pnt(p1_start.XYZ())
    closest_p1.BaryCenter(1 - t, p1_end, t)
    closest_p2 = gp_Pnt(p2_start.XYZ())
    closest_p2.BaryCenter(1 - s, p2_end, s)

    distance = closest_p1.Distance(closest_p2)
    if distance > max_distance:
        possible_intersect = False

    return t, s, possible_intersect


class IntersectType(TypedDict):
    parmOnCurve1: float
    parmOnCurve2: float
    point: gp_Pnt


def IntersectBSplines(
    curve1: Geom_BSplineCurve, curve2: Geom_BSplineCurve, tolerance: float = 1e-5
) -> list[IntersectType]:
    hulls = getRangesOfIntersection(curve1, curve2, tolerance)

    curve1_ints: list[BoundingBox] = []
    curve2_ints: list[BoundingBox] = []
    for hull in hulls:
        curve1_ints.append(hull.b1)
        curve2_ints.append(hull.b2)

    curve1_ints.sort(key=lambda b: b.range.min)
    curve2_ints.sort(key=lambda b: b.range.min)

    unique_curve1_ints: list[BoundingBox] = []
    if curve1_ints:
        unique_curve1_ints.append(curve1_ints[0])
        for i in range(1, len(curve1_ints)):
            if not curve1_ints[i] == curve1_ints[i - 1]:
                unique_curve1_ints.append(curve1_ints[i])
    curve1_ints = unique_curve1_ints

    unique_curve2_ints: list[BoundingBox] = []
    if curve2_ints:
        unique_curve2_ints.append(curve2_ints[0])
        for i in range(1, len(curve2_ints)):
            if not curve2_ints[i] == curve2_ints[i - 1]:
                unique_curve2_ints.append(curve2_ints[i])
    curve2_ints = unique_curve2_ints

    def is_adjacent(
        avoid_u_values: list[float], b1: BoundingBox, b2: BoundingBox
    ) -> bool:
        EPS = 1e-15
        if (
            len(avoid_u_values) > 0
            and min(
                [
                    min(abs(b1.range.max - u), abs(b2.range.min - u))
                    for u in avoid_u_values
                ]
            )
            < EPS
        ):
            return False
        return abs(b1.range.max - b2.range.min) < EPS

    def create_avoid_u_values(curve: Geom_BSplineCurve):
        """
        We want to avoid merge adjacent boxes if the merge creates a kink
        """
        avoid_u_values: list[float] = []
        for i in range(2, curve.NbKnots()):
            if curve.Multiplicity(i) > 1:
                avoid_u_values.append(curve.Knot(i))
        return avoid_u_values

    replace_adjacent_in_list(
        create_avoid_u_values(curve1), curve1_ints, is_adjacent, merge_boxes
    )
    replace_adjacent_in_list(
        create_avoid_u_values(curve2), curve2_ints, is_adjacent, merge_boxes
    )

    intersectionCandidates: list[BoundingBoxPair] = []
    for b1 in curve1_ints:
        for b2 in curve2_ints:
            if b1.Intersects(b2, tolerance):
                intersectionCandidates.append(BoundingBoxPair(b1, b2))

    results: list[IntersectType] = []

    def append_result(results: list[IntersectType], result: IntersectType):
        for r in results:
            if (
                abs(r["parmOnCurve1"] - result["parmOnCurve1"]) < tolerance
                and abs(r["parmOnCurve2"] - result["parmOnCurve2"]) < tolerance
            ):
                return
        results.append(result)

    _tolerance_distance = max(1e-10, min(1e-5, tolerance))

    for boxes in intersectionCandidates:
        c1 = BSplineAlgorithms.trimCurve(curve1, boxes.b1.range.min, boxes.b1.range.max)
        c2 = BSplineAlgorithms.trimCurve(curve2, boxes.b2.range.min, boxes.b2.range.max)

        # When the intersection is at the endpoints, it is hard for 2d minimizer to converge.
        # So we check the endpoints first before using the 2d minimizer
        # C++ code does not have this checking.

        def get_mid_point(u: float, v: float):
            p1 = c1.Value(u)
            p2 = c2.Value(v)
            p1.BaryCenter(1.0, p2, 1.0)
            return p1

        # check if cx1.Value(param1) is on cx2
        def check_point_on_curve2(
            cx1: Geom_BSplineCurve, param1: float, cx2: Geom_BSplineCurve
        ):
            v = cx2.FirstParameter()
            cx2_first_point = cx2.Value(v)
            if cx1.Value(param1).Distance(cx2_first_point) < _tolerance_distance:
                return param1, v

            v = cx2.LastParameter()
            cx2_last_point = cx2.Value(v)
            if cx1.Value(param1).Distance(cx2_last_point) < _tolerance_distance:
                return param1, v

            t, possible_intersect = is_point_on_line_segment(
                cx1.Value(param1),
                cx2_first_point,
                cx2_last_point,
                max(curvature(cx1), curvature(cx2)),
            )
            if not possible_intersect:
                return None, None

            res = minimize_scalar(
                lambda v: cx1.Value(param1).SquareDistance(cx2.Value(v)),
                bounds=(cx2.FirstParameter(), cx2.LastParameter()),
                method="Bounded",
                options={"xatol": 0.1 * _tolerance_distance * _tolerance_distance},
            )
            if res.success and res.fun < _tolerance_distance * _tolerance_distance:  # type: ignore
                return param1, res.x  # type: ignore
            else:
                return None, None

        u, v = check_point_on_curve2(c1, c1.FirstParameter(), c2)
        if u is not None and v is not None:
            result: IntersectType = {
                "parmOnCurve1": u,
                "parmOnCurve2": v,
                "point": get_mid_point(u, v),  # c1.Value(c1.FirstParameter())
            }
            append_result(results, result)
            continue

        u, v = check_point_on_curve2(c1, c1.LastParameter(), c2)
        if u is not None and v is not None:
            result: IntersectType = {
                "parmOnCurve1": u,
                "parmOnCurve2": v,
                "point": get_mid_point(u, v),  # c1.Value(c1.LastParameter())
            }
            append_result(results, result)
            continue

        v, u = check_point_on_curve2(c2, c2.FirstParameter(), c1)
        if u is not None and v is not None:
            result: IntersectType = {
                "parmOnCurve1": u,
                "parmOnCurve2": v,
                "point": get_mid_point(u, v),  # c2.Value(c2.FirstParameter())
            }
            append_result(results, result)
            continue

        v, u = check_point_on_curve2(c2, c2.LastParameter(), c1)
        if u is not None and v is not None:
            result: IntersectType = {
                "parmOnCurve1": u,
                "parmOnCurve2": v,
                "point": get_mid_point(u, v),  # c2.Value(c2.LastParameter())
            }
            append_result(results, result)
            continue

        # C++ code starts from here. Hence, it only uses 2d minimizer without first checking intersection at endpoints
        obj = CurveCurveDistanceObjective(c1, c2)

        # Try to find a good initial guess using straight-line analytic solution
        p1_start = c1.Value(c1.FirstParameter())
        p1_end = c1.Value(c1.LastParameter())
        p2_start = c2.Value(c2.FirstParameter())
        p2_end = c2.Value(c2.LastParameter())

        t_closest, s_closest, possible_intersect = line_line_intersection_3d(
            p1_start,
            p1_end,
            p2_start,
            p2_end,
            max(curvature(c1), curvature(c2)),
        )

        if not possible_intersect:
            continue

        # Use custom math_Vector class for optimizer inputs
        guess = math_Vector(1, 2)
        guess.SetValue(1, t_closest)
        guess.SetValue(2, s_closest)

        # C++ uses 1e-10 as the aTolenrance for math_FRPR which seems to be an error
        for i in range(2):
            converged = math_BFGS(obj, guess, _tolerance_distance * _tolerance_distance)
            if converged:
                break
            # print(f"Warning: `IntersectBSplines` not converge {i+1} time")
            # print(f'boxes.b1.range=[{boxes.b1.range.min}, {boxes.b1.range.max}]')
            # print(f'boxes.b2.range=[{boxes.b2.range.min}, {boxes.b2.range.max}]')
            # print(f'tolerance={tolerance}')

        u = obj.getUParam(guess.Value(1))
        v = obj.getVParam(guess.Value(2))

        if (
            u < min(c1.FirstParameter(), c1.LastParameter()) - _tolerance_distance
            or u > max(c1.FirstParameter(), c1.LastParameter()) + _tolerance_distance
        ):
            continue

        if (
            v < min(c2.FirstParameter(), c2.LastParameter()) - _tolerance_distance
            or v > max(c2.FirstParameter(), c2.LastParameter()) + _tolerance_distance
        ):
            continue

        distance = c1.Value(u).Distance(c2.Value(v))
        # print(f'distance={distance}, _tolerance_distance={_tolerance_distance}, tolerance={tolerance}')

        if distance < tolerance:
            result: IntersectType = {
                "parmOnCurve1": u,
                "parmOnCurve2": v,
                "point": get_mid_point(u, v),
            }
            append_result(results, result)

    return results


# --- Main function call example (for testing purposes) ---
if __name__ == "__main__":
    pass
