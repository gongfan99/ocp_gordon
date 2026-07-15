"""
Gordon curve interpolation in Python for CadQuery's OCP.

This module provides Python implementation of the Gordon curve interpolation
algorithm originally designed for OpenCASCADE, compatible with CadQuery's OCP.
"""
from importlib.metadata import version, PackageNotFoundError
from .internal.interpolate_curve_network import interpolate_curve_network, interpolate_curve_network_debug

try:
    __version__ = version("ocp-gordon")
except PackageNotFoundError:
    __version__ = "unknown version"

__all__ = ['interpolate_curve_network', 'interpolate_curve_network_debug', '__version__']
