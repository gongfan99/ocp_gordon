"""
Simple example demonstrating Gordon curve interpolation with build123d.

This example shows how to use the Gordon surface interpolation
functionality with build123d's OCP integration.
"""
# %%
from typing import Union, List
from build123d import *  # type: ignore
import numpy as np
from ocp_vscode import show, Camera, set_defaults

# set_defaults(reset_camera=Camera.CENTER, helper_scale=5)
set_defaults(reset_camera=Camera.KEEP, helper_scale=5)

from build123d_face_ext import Face_ext

# %%

# simple sine wave surface
def create_test_curves1():
    """
    Create test curves that form a proper intersecting network for Gordon surface interpolation.
    
    Returns:
        Tuple of (profiles, guides) - lists of B-spline curves that properly intersect
    """
    profiles: list[Edge] = []
    guides: list[Edge] = []
    
    # Define grid parameters
    num_profiles = 8
    num_guides = 8
    u_range = 5.0  # Range in u-direction (profiles)
    v_range = 8.0  # Range in v-direction (guides)
    
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

# aircraft engine cowling
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

# airliner fuselage
def create_test_curves3():
    profiles: list[Edge] = []
    guides: list[Edge] = []

    # Define points for the aircraft side profile in the YZ plane (X=0)
    # These points approximate the shape of an aircraft fuselage from nose to tail.
    # Y-coordinates are positive, Z-coordinates define the vertical profile.
    top_guide_points = [
        (0, 0, 1.47),
        (0, 1, 2.21),
        (0, 2.2, 2.94),
        (0, 3, 3.68),
        (0, 4, 4.56),
        (0, 5, 5.15),
        (0, 7, 5.70),
        (0, 10, 6.05),
        (0, 15, 6.25),
        (0, 30, 6.25),
        (0, 50, 6.21),
        (0, 60, 6.10),
        (0, 74.5, 4.96),
    ]

    bottom_guide_points = [
        (0, 0, -0.37),
        (0, 1, -0.92),
        (0, 2.2, -1.32),
        (0, 3, -1.54),
        (0, 4, -1.77),
        (0, 5, -1.91),
        (0, 7, -2.02),
        (0, 10, -2.10),
        (0, 15, -2.13),
        (0, 30, -2.10),
        (0, 48, -2.10),
        (0, 50, -2.06),
        (0, 52, -1.91),
        (0, 60, -0.55),
        (0, 68, 1.47),
        (0, 74.5, 3.49),
    ]

    guide1 = Spline(top_guide_points) # top guide
    guide2 = Spline(bottom_guide_points) # bottom guide
    
    # points = [Vector(*p) for p in top_guide_points]

    # create a circle passing p1 and p2 as diameter
    def circle_by_2_point(p1: Vector, p2: Vector):
        center = (p1 + p2) / 2
        loc1 = Location(center, (90 + Vector(0, 0, 1).get_signed_angle(p2 - p1), 0, 0))
        c1 = CenterArc(center=(0,0,0), radius=abs(p1-p2)/2, start_angle=0, arc_size=360)
        return c1.locate(loc1)

    profile_section_points = [0, 5, 10, 30, 50, top_guide_points[-1][1]]

    for section_point in profile_section_points:
        point1 = guide1.intersect(Plane((0, section_point, 0), z_dir=(0, -1, 0)))
        point2 = guide2.intersect(Plane((0, section_point, 0), z_dir=(0, -1, 0)))
        if point1 is not None and point2 is not None:
            vertex1 = point1.vertex()
            vertex2 = point2.vertex()
            if vertex1 is not None and vertex2 is not None:
                profiles.append(circle_by_2_point(vertex1.center(), vertex2.center()))

    guide_points = [0]
    guides = [guide1, guide2]
    for u in guide_points:
        guide3 = Spline([p@u for p in profiles])
        guide4 = guide3.mirror(Plane.YZ)
        guides.extend([guide3, guide4])
    
    # show(*points, *profiles, *guides)
    return profiles, guides

if __name__ == "__main__":
    """Main demonstration function."""
    print("Gordon Curve Interpolation Example")
    print("==================================")
    
    # Create test curves
    print("Creating test curves...")
    # profiles, guides = create_test_curves1()
    # profiles, guides = create_test_curves2()
    profiles, guides = create_test_curves3()
    # show(*profiles, *guides, reset_camera=Camera.KEEP)
    
    print(f"Created {len(profiles)} profile curves and {len(guides)} guide curves")
    
    # Perform Gordon interpolation
    print("Performing Gordon surface interpolation...")
    
    if 1:
        face10 = Face_ext.gordon_surface(
            profiles, guides, tolerance=3e-4
        )

        edge20 = (face10.edges() > Axis.Y)[0]
        face20 = Face.make_surface_patch([(edge20, face10, ContinuityLevel.C2)])

        edge30 = (face10.edges() < Axis.Y)[0]
        face30 = Face.make_surface_patch([(edge30, face10, ContinuityLevel.C0)])

        shell10 = Shell((face10, face20, face30))
        solid10 = Solid(shell10)

        show(solid10, *profiles, *guides)
        
