# special knowledge

## the mathematical rules for B-splines

- for any B-spline, `sum of multiplicities` = `Number of control points` + `degree` + 1
- for a clamped B-spline (a.k.a. open uniform B-spline), the first knot and the last knot have multiplicity of (degree + 1)
- for a periodic B-spline (a.k.a. closed B-spline), Knot vector arranged so first (degree) and last (degree) control points overlap cyclically
