import math
import sys
from typing import List, Tuple, Any, TypedDict
from scipy.optimize import minimize_scalar, OptimizeResult

from OCP.gp import gp_Pnt, gp_Vec
from OCP.Geom import Geom_Curve, Geom_BSplineCurve

from .misc import clone_bspline, math_Vector, Standard_Real, math_MultipleVarFunctionWithGradient, math_BFGS

# --- Helper functions and classes from C++ ---

def ReplaceAdjacentWithMerged(list_obj, is_adjacent, merged):
    it = 0
    while it < len(list_obj):
        nextIt = it + 1
        if nextIt >= len(list_obj):
            break

        if is_adjacent(list_obj[it], list_obj[nextIt]):
            merged_val = merged(list_obj[it], list_obj[nextIt])
            list_obj.pop(it)
            list_obj.pop(it)
            list_obj.insert(it, merged_val)
        else:
            it = nextIt

def maxval(v1, v2):
    return v1 if v1 > v2 else v2

def sqr(v):
    return v * v

def minCoords(p1: gp_Pnt, p2: gp_Pnt) -> gp_Pnt:
    result = gp_Pnt(p1.X(), p1.Y(), p1.Z())
    result.SetX(min(p1.X(), p2.X()))
    result.SetY(min(p1.Y(), p2.Y()))
    result.SetZ(min(p1.Z(), p2.Z()))
    return result

def maxCoords(p1: gp_Pnt, p2: gp_Pnt) -> gp_Pnt:
    result = gp_Pnt(p1.X(), p1.Y(), p1.Z())
    result.SetX(max(p1.X(), p2.X()))
    result.SetY(max(p1.Y(), p2.Y()))
    result.SetZ(max(p1.Z(), p2.Z()))
    return result

class Intervall:
    def __init__(self, mmin: float, mmax: float):
        self.min = mmin
        self.max = mmax

    def __eq__(self, other):
        EPS = 1e-15
        return abs(self.min - other.min) < EPS and abs(self.max - other.max) < EPS

class BoundingBox:
    def __init__(self, curve_or_other: Any = None):
        if isinstance(curve_or_other, Geom_BSplineCurve):
            curve = curve_or_other
            self.range = Intervall(curve.FirstParameter(), curve.LastParameter())
            self.low = gp_Pnt(math.inf, math.inf, math.inf)
            self.high = gp_Pnt(-math.inf, -math.inf, -math.inf)
            for i in range(1, curve.NbPoles() + 1):
                p = gp_Pnt(curve.Pole(i).XYZ())
                self.low = minCoords(self.low, p)
                self.high = maxCoords(self.high, p)
        elif isinstance(curve_or_other, BoundingBox):
            other = curve_or_other
            self.range = Intervall(other.range.min, other.range.max)
            self.low = gp_Pnt(other.low.XYZ())
            self.high = gp_Pnt(other.high.XYZ())
        else:
            self.range = Intervall(0.0, 0.0)
            self.low = gp_Pnt(0.0, 0.0, 0.0)
            self.high = gp_Pnt(0.0, 0.0, 0.0)

    def Intersects(self, other: 'BoundingBox', eps: float) -> bool:
        min_coords = maxCoords(self.low, other.low)
        max_coords = minCoords(self.high, other.high)
        return (min_coords.X() < max_coords.X() + eps) and \
               (min_coords.Y() < max_coords.Y() + eps) and \
               (min_coords.Z() < max_coords.Z() + eps)

    def Merge(self, other: 'BoundingBox') -> 'BoundingBox':
        self.range.max = other.range.max
        self.high = maxCoords(self.high, other.high)
        self.low = minCoords(self.low, other.low)
        return self

    def __eq__(self, other):
        return self.range == other.range

def curvature(curve: Geom_BSplineCurve) -> float:
    len_curve = curve.Pole(1).Distance(curve.Pole(curve.NbPoles()))
    total = 0.
    for i in range(1, curve.NbPoles()):
        p1 = curve.Pole(i)
        p2 = curve.Pole(i+1)
        dist = p1.Distance(p2)
        total += dist
    
    if abs(len_curve) < 1e-15:
        return 1.0
    return total / len_curve

class BoundingBoxPair:
    def __init__(self, b1: BoundingBox, b2: BoundingBox):
        self.b1 = b1
        self.b2 = b2

class BSplineAlgorithms:
    @staticmethod
    def trimCurve(curve: Geom_BSplineCurve, first: float, last: float) -> Geom_BSplineCurve:
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
        return True # The original C++ returns true, and F is an output parameter

    def Gradient(self, X: math_Vector, G: math_Vector) -> bool: # Gradient also takes G as an output parameter
        F_val = Standard_Real(0.0) # Create a temporary Standard_Real for F
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

        p1 = gp_Pnt(0,0,0)
        p2 = gp_Pnt(0,0,0)
        d1_vec = gp_Vec(0,0,0)
        d2_vec = gp_Vec(0,0,0)

        self.m_c1.D1(u, p1, d1_vec)
        self.m_c2.D1(v, p2, d2_vec)

        diff = gp_Vec(p1.X() - p2.X(), p1.Y() - p2.Y(), p1.Z() - p2.Z())
        F.value = diff.SquareMagnitude()
        
        G.SetValue(1, 2. * diff.Dot(d1_vec) * (self.m_c1.LastParameter() - self.m_c1.FirstParameter()) * self.d_getUParam(X.Value(1)))
        G.SetValue(2, -2. * diff.Dot(d2_vec) * (self.m_c2.LastParameter() - self.m_c2.FirstParameter()) * self.d_getVParam(X.Value(2)))

        return True

def getRangesOfIntersection(curve1: Geom_BSplineCurve, curve2: Geom_BSplineCurve, tolerance: float) -> List[BoundingBoxPair]:
    h1 = BoundingBox(curve1)
    h2 = BoundingBox(curve2)
    
    if not h1.Intersects(h2, tolerance):
        return []
    
    c1_curvature = curvature(curve1)
    c2_curvature = curvature(curve2)
    max_curvature = 1.0005
    
    if c1_curvature <= max_curvature and c2_curvature <= max_curvature:
        return [BoundingBoxPair(h1, h2)]
    
    curve1MidParm = 0.5 * (curve1.FirstParameter() + curve1.LastParameter())
    curve2MidParm = 0.5 * (curve2.FirstParameter() + curve2.LastParameter())
    
    results = []
    
    if c1_curvature > max_curvature and c2_curvature > max_curvature:
        c11 = BSplineAlgorithms.trimCurve(curve1, curve1.FirstParameter(), curve1MidParm)
        c12 = BSplineAlgorithms.trimCurve(curve1, curve1MidParm, curve1.LastParameter())
        
        c21 = BSplineAlgorithms.trimCurve(curve2, curve2.FirstParameter(), curve2MidParm)
        c22 = BSplineAlgorithms.trimCurve(curve2, curve2MidParm, curve2.LastParameter())
        
        results.extend(getRangesOfIntersection(c11, c21, tolerance))
        results.extend(getRangesOfIntersection(c11, c22, tolerance))
        results.extend(getRangesOfIntersection(c12, c21, tolerance))
        results.extend(getRangesOfIntersection(c12, c22, tolerance))
        
    elif c1_curvature <= max_curvature and max_curvature < c2_curvature:
        c21 = BSplineAlgorithms.trimCurve(curve2, curve2.FirstParameter(), curve2MidParm)
        c22 = BSplineAlgorithms.trimCurve(curve2, curve2MidParm, curve2.LastParameter())
        
        results.extend(getRangesOfIntersection(curve1, c21, tolerance))
        results.extend(getRangesOfIntersection(curve1, c22, tolerance))
        
    elif c2_curvature <= max_curvature and max_curvature < c1_curvature:
        c11 = BSplineAlgorithms.trimCurve(curve1, curve1.FirstParameter(), curve1MidParm)
        c12 = BSplineAlgorithms.trimCurve(curve1, curve1MidParm, curve1.LastParameter())
        
        results.extend(getRangesOfIntersection(c11, curve2, tolerance))
        results.extend(getRangesOfIntersection(c12, curve2, tolerance))
        
    return results

def replace_adjacent_in_list(list_obj, is_adjacent_func, merge_func):
    i = 0
    while i < len(list_obj):
        next_i = i + 1
        if next_i >= len(list_obj):
            break

        if is_adjacent_func(list_obj[i], list_obj[next_i]):
            merged_val = merge_func(list_obj[i], list_obj[next_i])
            list_obj.pop(i)
            list_obj.pop(i)
            list_obj.insert(i, merged_val)
        else:
            i = next_i

def merge_boxes(b1: BoundingBox, b2: BoundingBox) -> BoundingBox:
    new_box = BoundingBox(b1)
    new_box.Merge(b2)
    return new_box

class IntersectType(TypedDict):
    parmOnCurve1: float
    parmOnCurve2: float
    point: gp_Pnt

def IntersectBSplines(curve1: Geom_BSplineCurve, curve2: Geom_BSplineCurve, tolerance: float = 1e-5) -> List[IntersectType]:
    hulls = getRangesOfIntersection(curve1, curve2, tolerance)
    
    curve1_ints: List[BoundingBox] = []
    curve2_ints: List[BoundingBox] = []
    for hull in hulls:
        curve1_ints.append(hull.b1)
        curve2_ints.append(hull.b2)
    
    curve1_ints.sort(key=lambda b: b.range.min)
    curve2_ints.sort(key=lambda b: b.range.min)

    unique_curve1_ints: List[BoundingBox] = []
    if curve1_ints:
        unique_curve1_ints.append(curve1_ints[0])
        for i in range(1, len(curve1_ints)):
            if not curve1_ints[i] == curve1_ints[i-1]:
                unique_curve1_ints.append(curve1_ints[i])
    curve1_ints = unique_curve1_ints

    unique_curve2_ints: List[BoundingBox] = []
    if curve2_ints:
        unique_curve2_ints.append(curve2_ints[0])
        for i in range(1, len(curve2_ints)):
            if not curve2_ints[i] == curve2_ints[i-1]:
                unique_curve2_ints.append(curve2_ints[i])
    curve2_ints = unique_curve2_ints
    
    def is_adjacent(b1: BoundingBox, b2: BoundingBox) -> bool:
        EPS = 1e-15
        return abs(b1.range.max - b2.range.min) < EPS
    
    replace_adjacent_in_list(curve1_ints, is_adjacent, merge_boxes)
    replace_adjacent_in_list(curve2_ints, is_adjacent, merge_boxes)
    
    intersectionCandidates: List[BoundingBoxPair] = []
    for b1 in curve1_ints:
        for b2 in curve2_ints:
            if b1.Intersects(b2, tolerance):
                intersectionCandidates.append(BoundingBoxPair(b1, b2))

    results: List[IntersectType] = []

    for boxes in intersectionCandidates:
        c1 = BSplineAlgorithms.trimCurve(curve1, boxes.b1.range.min, boxes.b1.range.max)
        c2 = BSplineAlgorithms.trimCurve(curve2, boxes.b2.range.min, boxes.b2.range.max)

        _tolerance_distance = max(1e-10, min(1e-5, tolerance))

        # When the intersection is at the endpoints, it is hard for 2d minimizer to converge.
        # So we check the endpoints first before using the 2d minimizer
        # C++ code does not have this checking.        

        def get_mid_point(u: float, v: float):
            p1 = c1.Value(u)
            p2 = c2.Value(v)
            p1.BaryCenter(1.0, p2, 1.0)
            return p1
        
        # check if cx1.Value(param1) is on cx2
        def check_point_on_curve2(cx1: Geom_BSplineCurve, param1: float, cx2: Geom_BSplineCurve):
            v = cx2.FirstParameter()
            if cx1.Value(param1).Distance(cx2.Value(v)) < _tolerance_distance:
                return param1, v
            
            v = cx2.LastParameter()
            if cx1.Value(param1).Distance(cx2.Value(v)) < _tolerance_distance:
                return param1, v
        
            res = minimize_scalar(
                lambda v: cx1.Value(param1).SquareDistance(cx2.Value(v)),
                bounds=(cx2.FirstParameter(), cx2.LastParameter()),
                method="Bounded",
                options={"xatol": 0.1 * _tolerance_distance * _tolerance_distance},
            )
            if res.success and res.fun < _tolerance_distance * _tolerance_distance: # type: ignore
                return param1, res.x # type: ignore
            else:
                return None, None

        u, v = check_point_on_curve2(c1, c1.FirstParameter(), c2)
        if u is not None and v is not None:
            result: IntersectType = {
                "parmOnCurve1": u,
                "parmOnCurve2": v,
                "point": get_mid_point(u, v) # c1.Value(c1.FirstParameter())
            }
            results.append(result)
            continue

        u, v = check_point_on_curve2(c1, c1.LastParameter(), c2)
        if u is not None and v is not None:
            result: IntersectType = {
                "parmOnCurve1": u,
                "parmOnCurve2": v,
                "point": get_mid_point(u, v) # c1.Value(c1.LastParameter())
            }
            results.append(result)
            continue

        v, u = check_point_on_curve2(c2, c2.FirstParameter(), c1)
        if u is not None and v is not None:
            result: IntersectType = {
                "parmOnCurve1": u,
                "parmOnCurve2": v,
                "point": get_mid_point(u, v) # c2.Value(c2.FirstParameter())
            }
            results.append(result)
            continue

        v, u = check_point_on_curve2(c2, c2.LastParameter(), c1)
        if u is not None and v is not None:
            result: IntersectType = {
                "parmOnCurve1": u,
                "parmOnCurve2": v,
                "point": get_mid_point(u, v) # c2.Value(c2.LastParameter())
            }
            results.append(result)
            continue

        # C++ code starts from here. Hence, it only uses 2d minimizer without first checking intersection at endpoints
        obj = CurveCurveDistanceObjective(c1, c2)

        # Use custom math_Vector class for optimizer inputs
        guess = math_Vector(1, 2)
        guess.SetValue(1, 0.5)
        guess.SetValue(2, 0.5)

        # C++ uses 1e-10 as the aTolenrance for math_FRPR which seems to be an error
        converged = math_BFGS(obj, guess, _tolerance_distance * _tolerance_distance)

        if not converged:
            # print("Warning: `IntersectBSplines` not converge")
            # print(f'boxes.b1.range=[{boxes.b1.range.min}, {boxes.b1.range.max}], b2.range=[{boxes.b2.range.min}, {boxes.b2.range.max}]')
            # print(f'tolerance={tolerance}')
            continue

        u = obj.getUParam(guess.Value(1))
        v = obj.getVParam(guess.Value(2))

        if u < min(c1.FirstParameter(), c1.LastParameter()) - _tolerance_distance or u > max(c1.FirstParameter(), c1.LastParameter()) + _tolerance_distance:
            continue

        if v < min(c2.FirstParameter(), c2.LastParameter()) - _tolerance_distance or v > max(c2.FirstParameter(), c2.LastParameter()) + _tolerance_distance:
            continue

        distance = c1.Value(u).Distance(c2.Value(v))
        # print(f'distance={distance}, _tolerance_distance={_tolerance_distance}, tolerance={tolerance}')
        
        if distance < tolerance: # _tolerance_distance not work well
            result: IntersectType = {
                "parmOnCurve1": u,
                "parmOnCurve2": v,
                "point": get_mid_point(u, v)
            }
            results.append(result)

    
    return results

# --- Main function call example (for testing purposes) ---
if __name__ == '__main__':
    pass
