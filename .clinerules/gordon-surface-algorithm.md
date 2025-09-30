# B-spline Based Gordon Surface Creation Algorithm

This document serves as a reference to the Gordon Surface Algorithm.

[cite_start]This pseudocode provides a detailed implementation of **Algorithm 1** from the paper "TiGL - An Open Source Computational Geometry Library for Parametric Aircraft Design"[cite: 2]. It outlines the process for creating a single B-spline surface that interpolates a network of profile and guide curves.

## Algorithm Overview

The core challenge is that the input curves are arbitrarily parametrized. To use the Gordon surface formula, the curve network must be **compatible**, meaning profile curves must intersect guide curves at consistent parameter values. The algorithm first reparametrizes the curves to achieve this compatibility and then constructs the final surface by combining three intermediate surfaces.

---

### **Input**

- `original_profiles`: A list of N original profile B-spline curves, {˜f₁(˜u), ˜f₂(˜u), ..., ˜fₘ(˜u)}.
- `original_guides`: A list of M original guide B-spline curves, {˜g₁(˜v), ˜g₂(˜v), ..., ˜gₙ(˜v)}.

### **Output**

- `gordon_surface`: A single B-spline surface `s(u,v)` that interpolates the input curve network.

---

### **Procedure**

```pseudocode
FUNCTION create_gordon_surface(original_profiles, original_guides):

    // STEP 1: Compute intersection parameters for the original, incompatible network.
    // Each profile intersects each guide at a unique parameter on that curve.
    original_u_intersections[N][M] // Stores ˜u_k,l for each intersection
    original_v_intersections[N][M] // Stores ˜v_k,l for each intersection

    FOR l from 1 to M:
        FOR k from 1 to N:
            intersection_point = calculate_intersection(original_profiles[k], original_guides[l])
            original_u_intersections[k][l] = get_parameter_on_curve(original_profiles[k], intersection_point)
            original_v_intersections[k][l] = get_parameter_on_curve(original_guides[l], intersection_point)

    // STEPS 2-5: Calculate the new, consistent target parameters for the compatible network.
    // These are derived by averaging the original intersection parameters.
    target_u_params[M]
    target_v_params[N]

    FOR l from 1 to M: // For each guide
        target_u_params[l] = average(original_u_intersections[:][l]) // Average u-params along the guide

    FOR k from 1 to N: // For each profile
        target_v_params[k] = average(original_v_intersections[k][:]) // Average v-params along the profile

    // STEPS 6-7: Determine the number of control points for the new reparametrized curves.
    // Using the max count helps preserve the shape of the most complex input curve.
    n_profile_ctrl_pts = max_control_points(original_profiles)
    m_guide_ctrl_pts = max_control_points(original_guides)

    // STEPS 8-11: Reparametrize all curves to create a compatible network.
    // This uses a hybrid interpolation/approximation method (see paper Section 3.2.3).
    compatible_profiles[N]
    compatible_guides[M]

    FOR k from 1 to N: // Reparametrize each profile
        original_params = original_u_intersections[k][:] // The ˜u values for this profile
        compatible_profiles[k] = reparametrize_curve(
            curve = original_profiles[k],
            original_params = original_params,
            target_params = target_u_params,
            num_control_points = n_profile_ctrl_pts
        )

    FOR l from 1 to M: // Reparametrize each guide
        original_params = original_v_intersections[:][l] // The ˜v values for this guide
        compatible_guides[l] = reparametrize_curve(
            curve = original_guides[l],
            original_params = original_params,
            target_params = target_v_params,
            num_control_points = m_guide_ctrl_pts
        )

    // STEP 12: Compute the profile skinning surface Sf(u,v).
    // This surface interpolates the compatible profile curves at the target v-parameters.
    // This involves solving the linear system from Equation (15).
    surface_Sf = create_skinning_surface(
        curves_to_skin = compatible_profiles,
        interpolation_params = target_v_params
    )

    // STEP 13: Compute the guide skinning surface Sg(u,v).
    // This surface interpolates the compatible guide curves at the target u-parameters.
    // This involves solving the linear system from Equation (16).
    surface_Sg = create_skinning_surface(
        curves_to_skin = compatible_guides,
        interpolation_params = target_u_params
    )

    // STEP 14: Compute the tensor product surface T(u,v).
    // This surface interpolates the grid of intersection points.
    // This involves solving the linear system from Equation (17).
    intersection_points[N][M]
    FOR l from 1 to M:
        FOR k from 1 to N:
            intersection_points[k][l] = evaluate_curve(compatible_profiles[k], at_param=target_u_params[l])

    surface_T = create_tensor_product_surface(
        points = intersection_points,
        u_params = target_u_params,
        v_params = target_v_params
    )

    // STEP 15: Make the three intermediate surfaces compatible.
    // They must have the same degree and knot vectors in both u and v directions
    // before their control points can be combined.
    elevate_degree_and_insert_knots(surface_Sf, surface_Sg, surface_T)

    // STEP 16: Create the final Gordon surface by combining the control points.
    // s(u,v) = Sf(u,v) + Sg(u,v) - T(u,v)
    final_ctrl_pts = surface_Sf.control_points + surface_Sg.control_points - surface_T.control_points

    gordon_surface = create_bspline_surface(
        control_points = final_ctrl_pts,
        u_knots = surface_Sf.u_knots, // All knot vectors are now identical
        v_knots = surface_Sf.v_knots,
        u_degree = surface_Sf.u_degree,
        v_degree = surface_Sf.v_degree
    )

    // STEP 17: Return the final surface.
    RETURN gordon_surface
```
