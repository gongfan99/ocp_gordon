# Gordon Surface Library for CadQuery's OCP

This library provides a Python implementation for creating Gordon surfaces, a method for interpolating a network of curves to generate a smooth surface. It is designed to be compatible with CadQuery's OCP and leverages B-spline mathematics.

The implementation is written entirely in Python and is adapted from the original C++ code in [occ_gordon](https://github.com/rainman110/occ_gordon)

## Features

- Gordon surface interpolation from profile and guide curves.
- Compatibility with B-spline representations.
- Integration with CadQuery's OCP (OpenCASCADE Python) for geometric primitives.

## Installation

This package can be installed using pip.

```bash
pip install ocp_gordon
```

## Dependencies

- OCP (OpenCASCADE Python)
- NumPy
- SciPy

## Usage

Here's a basic example of how to use the library:

```python
# Assume you have profile_curves and guide_curves defined as lists of B-spline objects
# profile_curves = [...] # List of Geom_BSplineCurve objects
# guide_curves = [...]   # List of Geom_BSplineCurve objects

from ocp_gordon import interpolate_curve_network

gordon_surface = interpolate_curve_network(profile_curves, guide_curves, tolerance=3e-4)

```

For more detailed examples, please refer to the `examples/` directory in the source code.

## Test

To run tests, first install pytest, then:

```bash
python -m pytest

```

## Notable Difference from C++ Code

- In `intersect_bsplines.py`, the `math_BFGS` method is polyfilled and used in place of `math_FRPR`, as neither `math_BFGS` nor `math_FRPR` is usable in OCP due to the lack of `math_Vector` exposure. The intersect detection algorithm has been improved for both speed and reliability.
- In the `_solve()` function of `bspline_approx_interp.py`, regularization has been added to prevent singular matrix issues, which can occur in cases such as when the input curve is a B-spline derived from a circle.
- A new file, `misc.py`, has been introduced to implement missing OCP utilities. The primary additions include `clone_bspline` and `math_BFGS`.
- Modified `curve_network_sorter.py`, `bspline_algorithms.py`, and `interpolate_curve_network.py` to allow a single point to be used as either a profile or a guide.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Citing

The algorithm was originally described in:

[Siggel M. et. al. (2019), _TiGL: An Open Source Computational Geometry Library for Parametric Aircraft Design_](https://doi.org/10.1007/s11786-019-00401-y)

```
@article{siggel2019tigl,
	title={TiGL: an open source computational geometry library for parametric aircraft design},
	author={Siggel, Martin and Kleinert, Jan and Stollenwerk, Tobias and Maierl, Reinhold},
	journal={Mathematics in Computer Science},
	volume={13},
	number={3},
	pages={367--389},
	year={2019},
	publisher={Springer},
    doi={10.1007/s11786-019-00401-y}
}
```
