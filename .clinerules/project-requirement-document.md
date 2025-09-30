# Project Requirement Document: Gordon Curve Interpolation in Python for CadQuery

## 1. Introduction

This document outlines the requirements for converting the C++ implementation of the Gordon curve interpolation algorithm, originally designed for OpenCASCADE, into pure Python code compatible with CadQuery's OCP. The C++ code provides the `interpolate_curve_network` function, which takes a network of curves (profiles and guides) and generates a Gordon surface.

## 2. Core Algorithm: Gordon Surface Interpolation

The Gordon surface algorithm interpolates a network of curves to create a surface. The process involves several key stages:

### 2.1. Curve Network Preparation

- **Input:** A set of profile curves (typically in the u-direction) and guide curves (typically in the v-direction).
- **Compatibility:**
  - Input curves must be B-splines. If generic `Geom_Curve` types are provided, they need to be converted to `Geom_BSplineCurve`.
  - Curves must have compatible parameter ranges. Functions like `BSplineAlgorithms::matchParameterRange` and `BSplineAlgorithms::reparametrizeBSplineContinuouslyApprox` are used to ensure this.
  - A common knot vector is often required for combining curves, handled by `BSplineAlgorithms::createCommonKnotsVectorCurve`.
- **Intersection Calculation:** Determining intersection points between all profile and guide curves is crucial. The `IntersectBSplines` class implements a recursive subdivision and optimization-based approach for this.
- **Sorting:** Curves and their intersection parameters need to be sorted to establish a consistent order for surface construction.

### 2.2. Surface Construction

- **Skinning:** Intermediate surfaces are created by "skinning" the prepared curves.
  - `CurvesToSurface` is used to create a surface that interpolates the profile curves (skinning in the v-direction).
  - Similarly, it creates a surface that interpolates the guide curves (skinning in the u-direction).
- **Tensor Product Surface:** A surface is constructed from the grid of intersection points. This is handled by `PointsToBSplineInterpolation` and `BSplineAlgorithms::pointsToSurface`.
- **Gordon Surface Formula:** The final Gordon surface is computed by combining the intermediate surfaces using the formula:
  `S_Gordon(u,v) = S_Profiles(u,v) + S_Guides(u,v) - S_TensorProduct(u,v)`
  This operation is performed by manipulating the control points of the respective surfaces.

### 2.3. Continuity and Refinement

- **Continuity:** The algorithm aims to ensure C2 continuity for the resulting surface. This may involve knot manipulation techniques, as seen in `InterpolateCurveNetwork::EnsureC2`.
- **Closed Curves:** Special handling is required for curves that are closed (e.g., by duplicating control points and adjusting knot vectors) to ensure proper surface continuity.

## 3. Key C++ Components and Python Equivalents/Challenges

| C++ Component/Functionality    | Python Equivalent/Challenge                                                                                                                                                  |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `InterpolateCurveNetwork`      | Main orchestrator class. Python challenge: Recreating the complex state management and algorithm orchestration.                                                              |
| `BSplineAlgorithms`            | Utility class with static methods for B-spline manipulation. Python challenge: Implementing efficient B-spline algorithms without OpenCASCADE's native C++ performance.      |
| `IntersectBSplines`            | Computes intersections between B-spline curves using recursive subdivision. Python challenge: Implementing efficient intersection algorithms with proper tolerance handling. |
| `GordonSurfaceBuilder`         | Core Gordon surface construction. Python challenge: Implementing the Gordon formula with proper control point manipulation.                                                  |
| `CurvesToSurface`              | Surface skinning algorithm. Python challenge: Implementing efficient surface skinning with parameter compatibility.                                                          |
| `PointsToBSplineInterpolation` | Interpolates points to B-spline curves. Python challenge: Implementing the Park (2000) algorithm for node and knot selection.                                                |
| `CurveNetworkSorter`           | Orders and makes curve networks compatible. Python challenge: Implementing the complex curve network sorting and compatibility algorithms.                                   |
| `math_Matrix`                  | OpenCASCADE matrix class. Python challenge: Replacing with numpy arrays or similar Python matrix structures.                                                                 |
| `Handle(Geom_BSplineCurve)`    | OpenCASCADE smart pointers. Python challenge: Using Python's native object references with proper memory management.                                                         |
| `TColgp_Array1OfPnt`           | OpenCASCADE point arrays. Python challenge: Using Python lists or numpy arrays for point storage.                                                                            |

## 4. Python Implementation Strategy

### 4.1. Architecture Overview

The Python implementation should follow a similar modular structure:

- **Main Interface:** `interpolate_curve_network()` function similar to the C++ API
- **Core Classes:** Python classes mirroring the C++ architecture
- **Utility Functions:** Standalone functions for B-spline operations
- The generated Python code shall follow the C++ code line by line. The class names and member function names shall match one to one (python names will be lowercase with underscores).

### 4.2. Key Dependencies

- **CadQuery/OCP:** For B-spline curve and surface representations
- **NumPy:** For efficient matrix and array operations

### 4.3. Performance Considerations

- Use NumPy for vectorized operations where possible
- Implement critical algorithms in Cython if performance is insufficient
- Cache intermediate results to avoid redundant computations
- Profile and optimize intersection detection algorithms

## 5. Implementation Challenges and Solutions

### 5.1. B-spline Mathematics

- **Challenge:** Implementing complex B-spline algorithms (knot insertion, degree elevation, reparameterization)
- **Solution:** Leverage existing Python B-spline libraries where possible, implement core algorithms with careful testing

### 5.2. Intersection Detection

- **Challenge:** Efficiently finding intersections between B-spline curves
- **Solution:** Implement recursive subdivision with Newton-Raphson refinement, similar to C++ approach

### 5.3. Memory Management

- **Challenge:** Handling large curve networks and surfaces
- **Solution:** Use efficient data structures, implement lazy evaluation where appropriate

### 5.4. Numerical Stability

- **Challenge:** Maintaining numerical precision throughout the algorithm
- **Solution:** Use double precision throughout, implement robust tolerance handling

## 6. Testing and Validation

### 6.1. Unit Tests

- Test individual components (intersection detection, curve sorting, surface skinning)
- Compare results with C++ implementation for identical inputs
- Test edge cases (closed curves, degenerate cases, tolerance boundaries)

### 6.2. Integration Tests

- Test complete Gordon surface generation pipeline
- Validate against known test cases and expected results
- Performance benchmarking against C++ implementation

### 6.3. Validation Metrics

- Surface continuity verification (C0, C1, C2)
- Interpolation accuracy at curve intersections
- Computational performance metrics

## 7. Integration with CadQuery

### 7.1. API Design

- Follow CadQuery's Pythonic API conventions
- Provide both functional and object-oriented interfaces
- Support CadQuery's Workplane and geometric primitives

### 7.2. Data Conversion

- Convert between CadQuery geometry and internal representations
- Handle CadQuery's tolerance and precision settings
- Support CadQuery's coordinate system transformations

### 7.3. Error Handling

- Provide meaningful error messages for invalid inputs
- Handle tolerance violations and numerical issues gracefully
- Support CadQuery's exception handling patterns

## 8. Development Phases

### Phase 1: Core Infrastructure

- Basic B-spline utilities and data structures
- Curve intersection detection
- Simple test framework

### Phase 2: Algorithm Components

- Curve network sorting and compatibility
- Surface skinning implementation
- Gordon surface construction

### Phase 3: Integration and Optimization

- CadQuery integration
- Performance optimization
- Comprehensive testing

### Phase 4: Advanced Features

- Support for closed curves and periodic surfaces
- Enhanced continuity enforcement
- Additional interpolation methods

## 9. Success Criteria

- **Functional:** Correct Gordon surface generation for various curve networks
- **Performance:** Reasonable computational performance compared to C++ implementation
- **Robustness:** Handles edge cases and invalid inputs gracefully
- **Usability:** Clean API that integrates well with CadQuery workflows
- **Documentation:** Comprehensive documentation and examples

## 10. Risks and Mitigation

- **Performance Issues:** Profile early, consider Cython for critical sections
- **Numerical Stability:** Implement robust tolerance handling, test with various precisions
- **Algorithm Complexity:** Break down into manageable components, implement incrementally
- **OpenCASCADE Dependency:** Abstract OpenCASCADE-specific functionality for easier maintenance
