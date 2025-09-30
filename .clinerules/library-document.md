# Documentation for libraries

This document specifies which resources to use when the documentation or code snippet is required.

## OCP (OpenCASCADE python)

### Definitions for class/function/constant

- You have some memory about the OCP class definition, and you can use this memory to write the initial OCP code. However, before using any OCP class or member function, you _must_ use the `search_by_name` tool from the OCP-doc MCP server to verify its existence, signature, and usage. This is the _only_ way to be 100% sure about OCP class definitions and helps prevent errors caused by incorrect class names or usage.
- To check the existence and signature of a member method of an OCP class, you _must_ get the whole definition of the class.
- If there is an error in using a member method of an OCP class, you _must_ check the definition of the class.

### Several notes about OCP

- `Geom_BSplineCurve` does not have DownCast() function. Hence, to clone a bspline, instead of using `Geom_BSplineCurve.DownCast(bspline.Copy())`, you _must_ use `clone_bspline()` if available.
- `Geom_BSplineSurface` does not have DownCast() function. Hence, to clone a bspline surface, instead of using `Geom_BSplineSurface.DownCast(bspline_surface.Copy())`, you _must_ use `clone_bspline_surface()` if available.
- It has several array classes such as `TColStd_Array1OfInteger`, `TColStd_Array1OfReal`, `TColgp_Array1OfPnt` and `TColgp_Array2OfPnt`. To access an item with an index, you _must_ use `__call__()` instead of `Value()`. Note `SetValue()` should remain.
- math_Vector does not exist in OCP. Use Any in place of math_Vector.
- The set functions for math_Matrix are SetCol() and SetRow(). It does not have SetValue() function.
- If a function is a static method of a class, it has the `_s` suffix on the method names.

## OCCT (OpenCASCADE C++)

use context7 with the library id `/open-cascade-sas/occt`

## Build123d

use context7 with the library id `/gumyr/build123d`

## ocp_vscode

ocp_vscode is a library that can show build123d object. It has a function called show() whose document is in `https://github.com/bernhard-42/vscode-ocp-cad-viewer/blob/main/docs/show.md`
