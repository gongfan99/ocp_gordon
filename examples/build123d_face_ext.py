# %%
# This file extends the class build123d.Face to include gorden_surface()

import json
from pathlib import Path
from build123d import VectorLike, Vector, Edge, Face, ShapeList  # type: ignore
from OCP.Geom import Geom_Curve, Geom_BSplineCurve, Geom_BSplineSurface
from OCP.GeomAPI import GeomAPI_PointsToBSpline
from OCP.GeomConvert import GeomConvert
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
from OCP.BRep import BRep_Tool
from OCP.TopoDS import TopoDS_Face
from OCP.gp import gp_Pnt
from OCP.Precision import Precision

# Import our Gordon interpolation module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# from ocp_gordon import interpolate_curve_network, interpolate_curve_network_debug
from src_py.ocp_gordon import interpolate_curve_network, interpolate_curve_network_debug

# %%
def convert_bspline_to_edge(curve: Geom_Curve | Geom_BSplineCurve) -> Edge:
    make_edge = BRepBuilderAPI_MakeEdge(curve)
    ocp_edge = make_edge.Edge()
    return Edge(ocp_edge)

def convert_edge_to_bspline(edge: Edge) -> Geom_BSplineCurve:
    if edge.wrapped is None:
        raise ValueError("Edge cannot be empty")
    curve = BRep_Tool.Curve_s(edge.wrapped, 0, 1)
    return GeomConvert.CurveToBSplineCurve_s(curve)

def convert_bspline_surface_to_face(surface: Geom_BSplineSurface) -> Face:
    make_face = BRepBuilderAPI_MakeFace(surface, 1e-4) # Use a small tolerance
    ocp_face = make_face.Face()
    return Face(ocp_face)

def convert_face_to_bspline_surface(face: TopoDS_Face) -> Geom_BSplineSurface:
    """
    Converts a TopoDS_Face to a Geom_BSplineSurface.
    """
    # 1. Extract the underlying geometric surface from the TopoDS_Face
    geom_surface = BRep_Tool.Surface_s(face)

    # 2. Convert the Geom_Surface to a Geom_BSplineSurface
    bspline_surface = GeomConvert.SurfaceToBSplineSurface_s(geom_surface)

    return bspline_surface


class Face_ext(Face):
    """
    Extends build123d.Face with a gordon_surface method for interpolation.
    """

    @classmethod
    def gordon_surface(
        cls,
        profiles: list[Edge] | ShapeList[Edge],
        guides: list[Edge] | ShapeList[Edge],
        tolerance: float = 3e-4,
    ):
        """
        Creates a Gordon surface from a network of profile and guide curves.

        Args:
            profiles: A list or ShapeList of build123d.Edge objects representing profile curves.
            guides: A list or ShapeList of build123d.Edge objects representing guide curves.
            tolerance: Tolerance for surface creation and intersection calculations.

        Returns:
            A Face_ext object representing the interpolated Gordon surface.
        """
        ocp_profiles: list[Geom_Curve] = []
        ocp_guides: list[Geom_Curve] = []

        for edge in profiles:
            if edge.wrapped is None:
                raise ValueError("error")
            ocp_profiles.append(BRep_Tool.Curve_s(edge.wrapped, 0, 1))

        for edge in guides:
            if edge.wrapped is None:
                raise ValueError("error")
            ocp_guides.append(BRep_Tool.Curve_s(edge.wrapped, 0, 1))

        gordon_bspline_surface = interpolate_curve_network(
            ocp_profiles, ocp_guides, tolerance=tolerance
        )

        return cls(BRepBuilderAPI_MakeFace(gordon_bspline_surface, Precision.Confusion_s()).Face())

    # @classmethod
    # def gordon_surface_debug(
    #     cls,
    #     profiles: list[Edge] | ShapeList[Edge],
    #     guides: list[Edge] | ShapeList[Edge],
    #     tolerance: float = 1e-4,
    # ):
    #     """
    #     Creates a Gordon surface from a network of profile and guide curves.

    #     Args:
    #         profiles: A list or ShapeList of build123d.Edge objects representing profile curves.
    #         guides: A list or ShapeList of build123d.Edge objects representing guide curves.
    #         tolerance: Tolerance for surface creation and intersection calculations.

    #     Returns:
    #         A Face_ext object representing the interpolated Gordon surface.
    #     """
    #     ocp_profiles: list[Geom_Curve] = []
    #     ocp_guides: list[Geom_Curve] = []

    #     for edge in profiles:
    #         ocp_profiles.append(BRep_Tool.Curve_s(edge.wrapped, edge.param_at(0), edge.param_at(1)))

    #     for edge in guides:
    #         ocp_guides.append(BRep_Tool.Curve_s(edge.wrapped, edge.param_at(0), edge.param_at(1)))

    #     interpolator = interpolate_curve_network_debug(
    #         ocp_profiles, ocp_guides, tolerance=tolerance
    #     )

    #     # Create a Face from the OCP surface and then wrap it in Face_ext
    #     temp_face10 = convert_bspline_surface_to_face(interpolator.surface())
    #     temp_face20 = convert_bspline_surface_to_face(interpolator.surface_profiles())
    #     temp_face30 = convert_bspline_surface_to_face(interpolator.surface_guides())
    #     temp_face40 = convert_bspline_surface_to_face(interpolator.surface_intersections())
    #     return cls(temp_face10.wrapped), cls(temp_face20.wrapped), cls(temp_face30.wrapped), cls(temp_face40.wrapped)

    # def get_poles(self):
    #     ocp_surf: Geom_BSplineSurface = convert_face_to_bspline_surface(self.wrapped)
    #     poles: list[List[Vector]] = []
    #     for u in range(1, ocp_surf.NbUPoles()+1):
    #         poles.append([])
    #         for v in range(1, ocp_surf.NbVPoles()+1):
    #             pnt = ocp_surf.Pole(u, v)
    #             poles[-1].append(Vector(pnt))
    #     return poles
    
    # @staticmethod
    # def edge_to_spline(edge: Edge) -> Edge:
    #     curve = BRep_Tool.Curve_s(edge.wrapped, 0, 1)
    #     bspline = GeomConvert.CurveToBSplineCurve_s(curve)
    #     return convert_bspline_to_edge(bspline)


if __name__ == "__main__":
    pass
