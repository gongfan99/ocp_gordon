# special knowledge

## the mathematical rules for B-splines

**clamped B-spline (a.k.a. open B-spline)** the first knot and the last knot have multiplicity of (degree + 1), and `sum of multiplicities` = `Number of control points` + `degree` + 1
**periodic B-spline (a.k.a. closed B-spline)** Knot vector is arranged so first (degree) and last (degree) control points overlap cyclically. Hence, `sum of multiplicities` = `Number of control points` + 1
