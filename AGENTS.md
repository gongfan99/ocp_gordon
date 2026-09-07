# ocp_gordon

## Project

`ocp_gordon` is a Python implementation of Gordon-surface interpolation for CadQuery's OCP/OpenCASCADE geometry types.

- Python package: `src_py/ocp_gordon/`
- Tests: `tests/`
- Examples: `examples/`
- Separate C++ reference/source: `src_cpp/`
- Supported Python: `>=3.11, <3.15`
- Runtime dependencies: `cadquery_ocp_proxy`, NumPy, SciPy
- Test dependencies: `pytest`, `cadquery_ocp_novtk`

## Public API

```python
from ocp_gordon import interpolate_curve_network

surface = interpolate_curve_network(ucurves, vcurves, tolerance=3e-4)
```

`ucurves` and `vcurves` accept `Geom_Curve` or `Geom_BSplineCurve` objects. Generic curves are converted to B-splines. The function returns a `Geom_BSplineSurface`; `interpolate_curve_network_debug` returns the `InterpolateCurveNetwork` object for inspecting component surfaces and parameters.

Inputs must contain at least two unique curves in each direction, and every u/v pair must intersect within `tolerance`. The implementation normalizes curve parameters to `[0, 1]`, sorts the network, and reparameterizes it for compatibility. Interior zero-length curves and networks closed in both directions are rejected. One closed direction is supported by duplicating its boundary curve.

## Algorithm and code map

The pipeline is:

1. Convert curves to B-splines and normalize their parameter ranges.
2. Find pairwise intersections with `IntersectBSplines`.
3. Order curves and intersection parameters with `CurveNetworkSorter`.
4. Reparameterize curves with `BSplineAlgorithms`.
5. Build profile and guide skinning surfaces with `CurvesToSurface`.
6. Build the intersection/tensor-product surface with `PointsToBSplineInterpolation`.
7. Match degrees and knot vectors, then apply
   `S = S_profiles + S_guides - S_intersections` in `GordonSurfaceBuilder`.
8. Remove excess internal knot multiplicity in `InterpolateCurveNetwork` as the C2 post-processing step.

Important modules include:

- `bspline_algorithms.py`: conversion, reparameterization, knots, degree, closure, scaling, and surface helpers.
- `intersect_bsplines.py`: curve intersection detection and refinement.
- `curve_network_sorter.py`: network ordering and parameter ordering.
- `curves_to_surface.py`: compatible-curve skinning.
- `points_to_bspline_interpolation.py`: point-to-B-spline interpolation.
- `misc.py`: OCP cloning, optimizer/polyfill helpers, and serialization utilities.
- `error.py`: project-specific error codes and exception type.

## Development

Install the test environment with:

```bash
uv pip install .[test]
```

Run the suite with:

```bash
python -m pytest
```

Keep changes covered by focused unit tests and the full suite. Current tests exercise the public pipeline and internal B-spline, intersection, sorting, skinning, interpolation, and utility code.

## OCP and numerical conventions

- OCP 8 typed arrays come from `OCP.collections` (for example, `Array1_gp_Pnt`, `Array2_gp_Pnt`, `Array1_double`, and `Array1_int`); read with `.Value(...)` and write with `SetValue(...)`.
- Use `clone_bspline` and `clone_bspline_surface` from `misc.py` when copying OCP B-splines.
- Preserve double-precision numerical behavior and use the existing tolerance/scale helpers rather than introducing unrelated tolerances.
- Changes to parameterization, knot multiplicity, closure, or intersection logic must be validated against the relevant tests because small errors affect the final surface.

## B-spline rules

- Clamped/open B-splines have endpoint knot multiplicity `degree + 1`; total knot multiplicity is `control_points + degree + 1`.
- Periodic/closed B-splines use cyclically overlapping endpoint poles; total knot multiplicity is `control_points + 1`.
