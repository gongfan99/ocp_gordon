"""
Simple example demonstrating Gordon curve interpolation with build123d.

This example shows how to use the Gordon surface interpolation
functionality with build123d's OCP integration.
"""
# %%
from typing import Union, List
from build123d import *  # type: ignore
import math
from build123d import Shape
import numpy as np
from ocp_vscode import show, Camera

from build123d_face_ext import Face_ext

# %%

def create_test_curves1():
    """
    Create test curves that form a proper intersecting network for Gordon surface interpolation.
    
    Returns:
        Tuple of (profiles, guides) - lists of B-spline curves that properly intersect
    """
    profiles: list[Edge] = []
    guides: list[Edge] = []
    
    # Define grid parameters
    num_profiles = 3
    num_guides = 4
    u_range = 8.0  # Range in u-direction (profiles)
    v_range = 5.0  # Range in v-direction (guides)
    
    # Create intersection points grid
    # This defines where profiles and guides should intersect
    intersection_points = np.zeros((num_profiles, num_guides, 3))
    
    for i in range(num_profiles):
        for j in range(num_guides):
            # Create a grid of points with some variation for a more interesting surface
            u = i * u_range / (num_profiles - 1) if num_profiles > 1 else 0
            v = j * v_range / (num_guides - 1) if num_guides > 1 else 0
            
            # Add some 3D variation to make the surface more interesting
            z = 0.5 * np.sin(u * 0.5) * np.cos(v * 0.5)
            
            intersection_points[i, j] = [u, v, z]
    
    # Create profile curves (u-direction)
    for i in range(num_profiles):
        points: list[VectorLike] = []
        
        # Each profile curve goes through all guide intersection points at this profile index
        for j in range(num_guides):
            x, y, z = intersection_points[i, j]
            points.append((x, y, z))
        
        # Create B-spline curve through these points
        bspline_curve = Spline(points)
        profiles.append(bspline_curve)
    
    # Create guide curves (v-direction)
    for j in range(num_guides):
        points: list[VectorLike] = []
        
        # Each guide curve goes through all profile intersection points at this guide index
        for i in range(num_profiles):
            x, y, z = intersection_points[i, j]
            points.append((x, y, z))
        
        # Create B-spline curve through these points
        bspline_curve = Spline(points)
        guides.append(bspline_curve)
    
    return profiles, guides

def create_test_curves2():
    """
    Create test curves that form a proper intersecting network for Gordon surface interpolation.
    
    Returns:
        Tuple of (profiles, guides) - lists of B-spline curves that properly intersect
    """    
    # Define grid parameters
    radius = 10
    length = 36

    outer = Spline([(0.8, 1), (1.1, 0.35), (1.0, 0)])
    inner = Spline([(0.9, 0), (0.85, 0.35), (0.7, 1)])
    num_points = 40
    points = *[outer@(i/num_points) for i in range(num_points+1)], *[inner@(i/num_points) for i in range(num_points+1)]
    points = [Vector(p.X * radius, p.Y * length) for p in points]

    guide1 = Spline(points)
    guides: List[Edge] = [Rot(0,i*90) * guide1 for i in range(4)] # type: ignore
    # show(guide1, *points, reset_camera=Camera.KEEP)

    def to_circle(v: Vector) -> Edge:
        return Pos(0, v.Y) * Rot(90) * CenterArc((0,0,0), 1, 0, 360).scale(v.X) # type: ignore
    
    profiles = [to_circle(guide1@0), to_circle(guide1@1)]
    
    return profiles, guides


if __name__ == "__main__":
    """Main demonstration function."""
    print("Gordon Curve Interpolation Example")
    print("==================================")
    
    # Create test curves
    print("Creating test curves...")
    # profiles, guides = create_test_curves1()
    profiles, guides = create_test_curves2()
    show(*profiles, *guides, reset_camera=Camera.KEEP)
    
    print(f"Created {len(profiles)} profile curves and {len(guides)} guide curves")
    
    # Perform Gordon interpolation
    print("Performing Gordon surface interpolation...")
    
    if 1:
        face10 = Face_ext.gordon_surface(
            profiles, guides, tolerance=1e-4
        )

        print("Successfully created Gordon surface!")
        print("The resulting surface can be exported or used in further CAD operations.")
        
        # point01 = Vector(face10.position_at(0.8,0))

        show(face10, *profiles, *guides, alphas=[1], reset_camera=Camera.KEEP)
        
