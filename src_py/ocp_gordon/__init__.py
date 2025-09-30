"""
Gordon curve interpolation in Python for CadQuery's OCP.

This module provides Python implementation of the Gordon curve interpolation
algorithm originally designed for OpenCASCADE, compatible with CadQuery's OCP.
"""

from .internal.interpolate_curve_network import interpolate_curve_network, interpolate_curve_network_debug

__all__ = ['interpolate_curve_network', 'interpolate_curve_network_debug']
